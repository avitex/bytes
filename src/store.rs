use core::ops::RangeBounds;
use core::ptr::NonNull;
use core::{fmt, mem, ptr, slice, usize};

use alloc::{boxed::Box, vec::Vec};

use crate::loom::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};

const KIND_ARC: usize = 0b0;
const KIND_VEC: usize = 0b1;
const KIND_MASK: usize = 0b1;

/// Holds a owned, or shared reference to stored bytes and
/// a window into the data.
///
/// Will handle dropping the data if is the sole owner.
#[derive(Debug)]
pub(crate) struct Handle {
    /// The length of the window into the backing data.
    len: usize,
    /// Pointer to the start of the window into the backing data.
    ptr: NonNull<u8>,
    /// Shared data, specific over the backing storage.
    data: AtomicPtr<()>,
    /// Vtable used for base operations over handle.
    vtable: &'static Vtable,
}

impl Handle {
    #[inline]
    pub const fn new() -> Self {
        const EMPTY: &[u8] = &[];
        Self::from_static(EMPTY)
    }

    #[inline]
    pub const fn from_static(data: &'static [u8]) -> Self {
        let ptr = data.as_ptr() as *mut u8;
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        Self {
            ptr,
            len: 0,
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &STATIC_VTABLE,
        }
    }

    #[inline]
    pub fn from_vec(vec: Vec<u8>) -> (Handle, usize) {
        // `Vec::into_boxed_slice` doesn't return a heap allocation
        // for empty vectors, so the pointer isn't aligned enough for
        // the KIND_VEC stashing to work.
        if vec.is_empty() {
            return (Handle::new(), 0);
        }

        // Break the vec into it's raw components.
        // TODO: Consider `Vec::into_raw_parts`.
        let cap = vec.capacity();
        let slice = vec.into_boxed_slice();
        let len = slice.len();
        // We have check the vec to make sure it is not empty.
        let ptr = unsafe { non_null_ptr(slice.as_ptr() as _) };
        let data = ptr.as_ptr() as usize;

        let handle = if data & 0b1 == 0 {
            let data = data | KIND_VEC;
            Self {
                ptr,
                len,
                data: AtomicPtr::new(data as *mut _),
                vtable: &PROMOTABLE_EVEN_VTABLE,
            }
        } else {
            Self {
                ptr,
                len,
                data: AtomicPtr::new(data as *mut _),
                vtable: &PROMOTABLE_ODD_VTABLE,
            }
        };

        (handle, cap)
    }

    #[inline]
    pub fn is_window_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn window_len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn window_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    unsafe fn window_ptr_offset(&self, offset: isize) -> NonNull<u8> {
        non_null_ptr(self.ptr.as_ptr().offset(offset))
    }

    #[inline]
    pub fn window_split_off(&mut self, at: usize) -> Self {
        assert!(
            at <= self.window_len(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.window_len(),
        );

        if at == self.window_len() {
            return Self::new();
        }

        if at == 0 {
            return mem::replace(self, Self::new());
        }

        let mut ret = self.clone();

        self.len = at;

        unsafe { ret.window_inc_start(at) };

        ret
    }

    #[inline]
    pub fn window_slice(&self, range: impl RangeBounds<usize>) -> Self {
        use core::ops::Bound;

        let len = self.window_len();

        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        assert!(
            begin <= end,
            "range start must not be greater than end: {:?} <= {:?}",
            begin,
            end,
        );
        assert!(
            end <= len,
            "range end out of bounds: {:?} <= {:?}",
            end,
            len,
        );

        if end == begin {
            return Self::new();
        }

        let mut ret = self.clone();

        ret.len = end - begin;
        ret.ptr = unsafe { self.window_ptr_offset(begin as isize) };

        ret
    }

    #[inline]
    pub unsafe fn window_inc_start(&mut self, by: usize) {
        // should already be asserted, but debug assert for tests
        debug_assert!(self.len >= by, "internal: inc_start out of bounds");
        self.len -= by;
        self.ptr = self.window_ptr_offset(by as isize);
        unimplemented!()
    }

    #[inline]
    pub unsafe fn promote_to_shared(&mut self) {
        unimplemented!()
    }

    #[inline]
    pub unsafe fn get_windowed_slice(&self) -> &[u8] {
        slice::from_raw_parts(self.ptr.as_ptr(), self.len)
    }

    #[inline]
    pub unsafe fn get_windowed_slice_mut(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
    }

    // #[inline]
    // pub unsafe fn get_data_mut<T>(&self) -> &mut *mut T {
    //     &mut (*self.data.get_mut() as _)
    // }

    #[inline]
    pub unsafe fn into_vec(mut self) -> Vec<u8> {
        // We uniquely own the buffer, so it's ok to call `into_vec`
        let vec = (self.vtable.into_vec)(&mut self.data, self.ptr, self.len);
        // We take ownership of the data ptr, so no further deconstruction is needed.
        mem::forget(self);
        // Return the vec
        vec
    }
}

impl Clone for Handle {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { (self.vtable.clone)(&self.data, self.ptr, self.len) }
    }
}

impl Drop for Handle {
    #[inline]
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(&mut self.data, self.ptr, self.len) }
    }
}

pub(crate) struct Vtable {
    /// fn(data, ptr, len)
    pub clone: unsafe fn(&AtomicPtr<()>, NonNull<u8>, usize) -> Handle,
    /// fn(data, ptr, len)
    pub drop: unsafe fn(&mut AtomicPtr<()>, NonNull<u8>, usize),
    /// fn(data, ptr, len)
    pub into_vec: unsafe fn(&mut AtomicPtr<()>, NonNull<u8>, usize) -> Vec<u8>,
}

impl fmt::Debug for Vtable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vtable")
            .field("clone", &(self.clone as *const ()))
            .field("drop", &(self.drop as *const ()))
            .field("into_vec", &(self.into_vec as *const ()))
            .finish()
    }
}

// ===== impl StaticVtable =====

const STATIC_VTABLE: Vtable = Vtable {
    clone: static_clone,
    drop: static_drop,
    into_vec: static_into_vec,
};

unsafe fn static_clone(_: &AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Handle {
    Handle::from_static(slice::from_raw_parts(ptr.as_ptr(), len))
}

unsafe fn static_drop(_: &mut AtomicPtr<()>, _: NonNull<u8>, _: usize) {
    // nothing to drop for &'static [u8]
}

unsafe fn static_into_vec(_: &mut AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Vec<u8> {
    let slice = slice::from_raw_parts(ptr.as_ptr(), len);
    let mut buf = Vec::with_capacity(len);
    buf.extend_from_slice(slice);
    buf
}

// ===== impl PromotableVtable =====

static PROMOTABLE_EVEN_VTABLE: Vtable = Vtable {
    clone: promotable_even_clone,
    drop: promotable_even_drop,
    into_vec: promotable_even_into_vec,
};

static PROMOTABLE_ODD_VTABLE: Vtable = Vtable {
    clone: promotable_odd_clone,
    drop: promotable_odd_drop,
    into_vec: promotable_odd_into_vec,
};

unsafe fn promotable_even_clone(data: &AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Handle {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shallow_clone_arc(shared as _, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        let buf = (shared as usize & !KIND_MASK) as *mut u8;
        shallow_clone_vec(data, shared, buf, ptr, len)
    }
}

unsafe fn promotable_even_drop(data: &mut AtomicPtr<()>, ptr: NonNull<u8>, len: usize) {
    let shared = *data.get_mut();
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        release_shared(shared as *mut Shared);
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        let buf = (shared as usize & !KIND_MASK) as *mut u8;
        drop(rebuild_boxed_slice(buf, ptr, len));
    }
}

unsafe fn promotable_even_into_vec(
    data: &mut AtomicPtr<()>,
    ptr: NonNull<u8>,
    len: usize,
) -> Vec<u8> {
    let shared = *data.get_mut();
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        release_shared_into_vec(shared as *mut Shared, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        let buf = (shared as usize & !KIND_MASK) as *mut u8;
        rebuild_boxed_slice(buf, ptr, len).into_vec()
    }
}

unsafe fn promotable_odd_clone(data: &AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Handle {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shallow_clone_arc(shared as _, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        shallow_clone_vec(data, shared, shared as *mut u8, ptr, len)
    }
}

unsafe fn promotable_odd_drop(data: &mut AtomicPtr<()>, ptr: NonNull<u8>, len: usize) {
    let shared = *data.get_mut();
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        release_shared(shared as *mut Shared);
    } else {
        debug_assert_eq!(kind, KIND_VEC);

        drop(rebuild_boxed_slice(shared as *mut u8, ptr, len));
    }
}

unsafe fn promotable_odd_into_vec(
    data: &mut AtomicPtr<()>,
    ptr: NonNull<u8>,
    len: usize,
) -> Vec<u8> {
    let shared = *data.get_mut();
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        release_shared_into_vec(shared as *mut Shared, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);

        rebuild_boxed_slice(shared as *mut u8, ptr, len).into_vec()
    }
}

// ===== impl SharedVtable =====

struct Shared {
    // holds vec for drop, but otherwise doesnt access it
    vec: Vec<u8>,
    ref_cnt: AtomicUsize,
}

// Assert that the alignment of `Shared` is divisible by 2.
// This is a necessary invariant since we depend on allocating `Shared` a
// shared object to implicitly carry the `KIND_ARC` flag in its pointer.
// This flag is set when the LSB is 0.
const _: [(); 0 - mem::align_of::<Shared>() % 2] = []; // Assert that the alignment of `Shared` is divisible by 2.

static SHARED_VTABLE: Vtable = Vtable {
    clone: shared_clone,
    drop: shared_drop,
    into_vec: shared_into_vec,
};

unsafe fn shared_clone(data: &AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Handle {
    let shared = data.load(Ordering::Acquire);
    shallow_clone_arc(shared as _, ptr, len)
}

unsafe fn shared_drop(data: &mut AtomicPtr<()>, _ptr: NonNull<u8>, _len: usize) {
    let shared = *data.get_mut();
    release_shared(shared as *mut Shared);
}

unsafe fn shared_into_vec(data: &mut AtomicPtr<()>, ptr: NonNull<u8>, len: usize) -> Vec<u8> {
    let shared = *data.get_mut();
    release_shared_into_vec(shared as *mut Shared, ptr, len)
}

unsafe fn shallow_clone_arc(shared: *mut Shared, ptr: NonNull<u8>, len: usize) -> Handle {
    let old_size = (*shared).ref_cnt.fetch_add(1, Ordering::Relaxed);

    if old_size > usize::MAX >> 1 {
        crate::abort();
    }

    Handle {
        ptr,
        len,
        data: AtomicPtr::new(shared as _),
        vtable: &SHARED_VTABLE,
    }
}

#[cold]
unsafe fn shallow_clone_vec(
    atom: &AtomicPtr<()>,
    ptr: *const (),
    buf: *mut u8,
    offset: NonNull<u8>,
    len: usize,
) -> Handle {
    // If  the buffer is still tracked in a `Vec<u8>`. It is time to
    // promote the vec to an `Arc`. This could potentially be called
    // concurrently, so some care must be taken.

    // First, allocate a new `Shared` instance containing the
    // `Vec` fields. It's important to note that `ptr`, `len`,
    // and `cap` cannot be mutated without having `&mut self`.
    // This means that these fields will not be concurrently
    // updated and since the buffer hasn't been promoted to an
    // `Arc`, those three fields still are the components of the
    // vector.
    let vec = rebuild_boxed_slice(buf, offset, len).into_vec();
    let shared = Box::new(Shared {
        vec,
        // Initialize refcount to 2. One for this reference, and one
        // for the new clone that will be returned from
        // `shallow_clone`.
        ref_cnt: AtomicUsize::new(2),
    });

    let shared = Box::into_raw(shared);

    // The pointer should be aligned, so this assert should
    // always succeed.
    debug_assert!(
        0 == (shared as usize & KIND_MASK),
        "internal: Box<Shared> should have an aligned pointer",
    );

    // Try compare & swapping the pointer into the `arc` field.
    // `Release` is used synchronize with other threads that
    // will load the `arc` field.
    //
    // If the `compare_and_swap` fails, then the thread lost the
    // race to promote the buffer to shared. The `Acquire`
    // ordering will synchronize with the `compare_and_swap`
    // that happened in the other thread and the `Shared`
    // pointed to by `actual` will be visible.
    let actual = atom.compare_and_swap(ptr as _, shared as _, Ordering::AcqRel);

    if actual as usize == ptr as usize {
        // The upgrade was successful, the new handle can be
        // returned.
        return Handle {
            ptr: offset,
            len,
            data: AtomicPtr::new(shared as _),
            vtable: &SHARED_VTABLE,
        };
    }

    // The upgrade failed, a concurrent clone happened. Release
    // the allocation that was made in this thread, it will not
    // be needed.
    let shared = Box::from_raw(shared);
    mem::forget(*shared);

    // Buffer already promoted to shared storage, so increment ref
    // count.
    shallow_clone_arc(actual as _, offset, len)
}

unsafe fn release_shared(ptr: *mut Shared) {
    // `Shared` storage... follow the drop steps from Arc.
    if (*ptr).ref_cnt.fetch_sub(1, Ordering::Release) != 1 {
        return;
    }

    // This fence is needed to prevent reordering of use of the data and
    // deletion of the data. Because it is marked `Release`, the decreasing
    // of the reference count synchronizes with this `Acquire` fence. This
    // means that use of the data happens before decreasing the reference
    // count, which happens before this fence, which happens before the
    // deletion of the data.
    //
    // As explained in the [Boost documentation][1],
    //
    // > It is important to enforce any possible access to the object in one
    // > thread (through an existing reference) to *happen before* deleting
    // > the object in a different thread. This is achieved by a "release"
    // > operation after dropping a reference (any access to the object
    // > through this reference must obviously happened before), and an
    // > "acquire" operation before deleting the object.
    //
    // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
    atomic::fence(Ordering::Acquire);

    // Drop the data
    Box::from_raw(ptr);
}

unsafe fn release_shared_into_vec(
    data_ptr: *mut Shared,
    offset: NonNull<u8>,
    len: usize,
) -> Vec<u8> {
    // `Shared` storage... follow the drop steps from Arc.
    let ref_cnt = (*data_ptr).ref_cnt.fetch_sub(1, Ordering::Release);

    // See the comment in `release_shared` for the reasoning.
    atomic::fence(Ordering::Acquire);

    if ref_cnt != 1 {
        // We wish extract the vec, but there are other shared references
        // meaning we have to clone the data.
        return slice::from_raw_parts(offset.as_ptr(), len).to_vec();
    }

    // Re-construct the shared box.
    let mut shared = Box::from_raw(data_ptr);

    // We want to extract the vec data from the box, so we
    // first create an empty vec to swap with the one we care about.
    let mut vec = Vec::new();

    // Swap the empty vec, and the vec with the data.
    mem::swap(&mut vec, &mut shared.vec);

    // Drop the shared data.
    drop(shared);

    // Because bytes is a window into the vec, we need to
    // calculate the relative offset into the vec we are
    // pointing to.
    let rel_offset = offset.as_ptr() as usize - vec.as_ptr() as usize;

    // Drop any data at the end and start of the vec we don't care about.
    vec.truncate(rel_offset + len);
    vec.drain(..rel_offset);

    // Return the vec
    vec
}

unsafe fn rebuild_boxed_slice(buf: *mut u8, offset: NonNull<u8>, len: usize) -> Box<[u8]> {
    let cap = (offset.as_ptr() as usize - buf as usize) + len;
    Box::from_raw(slice::from_raw_parts_mut(buf, cap))
}

unsafe fn non_null_ptr(ptr: *mut u8) -> NonNull<u8> {
    if cfg!(debug_assertions) {
        NonNull::new(ptr).expect("pointer should be non-null")
    } else {
        NonNull::new_unchecked(ptr)
    }
}
