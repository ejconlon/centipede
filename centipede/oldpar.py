# class LockBox[T]:
#     def __init__(self, value: T):
#         self._lock = Condition()
#         self._box = Box(value)
#
#     def __enter__(self) -> Box[T]:
#         self._lock.acquire()
#         return self._box
#
#     def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pyright: ignore
#         self._lock.release()
#
#
# def is_fatal_exception(exc: Exception):
#     return isinstance(exc, SystemExit) or isinstance(exc, KeyboardInterrupt)
#
#
# @dataclass(frozen=True, eq=False)
# class CleanupError(Exception):
#     cleanup_error: Exception
#     run_error: Optional[Exception]
#
#
# @dataclass(frozen=True, eq=False)
# class EventChain:
#     _parent: Optional[EventChain]
#     _event: Event
#
#     @staticmethod
#     def root() -> EventChain:
#         return EventChain(_parent=None, _event=Event())
#
#     def child(self) -> EventChain:
#         sub = Event()
#         if self._event.is_set():
#             sub.set()
#         return EventChain(_parent=self, _event=sub)
#
#     def is_set(self) -> bool:
#         if self._event.is_set():
#             return True
#         else:
#             head = self._parent
#             steps = 0
#             while head is not None:
#                 if head.is_set():
#                     self._finalize(steps)
#                     return True
#                 else:
#                     head = head._parent
#                     steps += 1
#             return False
#
#     def _finalize(self, steps: int) -> None:
#         self._event.set()
#         head = self._parent
#         while head is not None:
#             if steps == 0:
#                 break
#             else:
#                 head._event.set()
#                 head = head._parent
#                 steps -= 1
#
#     def set(self) -> None:
#         if not self.is_set():
#             self._event.set()
#
#     def wait(self) -> None:
#         if not self.is_set():
#             self._event.wait()
#
#
# class Tag(Enum):
#     Empty = 0
#     Throw = 1
#     Resolve = 2
#
#
# @dataclass(frozen=True)
# class Tagged[T]:
#     _tag: Tag
#     _union: Union[None, Exception, T]
#
#     @staticmethod
#     def empty() -> Tagged[T]:
#         return Tagged(_tag=Tag.Empty, _union=None)
#
#     @staticmethod
#     def throw(exc: Exception) -> Tagged[T]:
#         return Tagged(_tag=Tag.Throw, _union=exc)
#
#     @staticmethod
#     def resolve(value: T) -> Tagged[T]:
#         return Tagged(_tag=Tag.Resolve, _union=value)
#
#     def is_empty(self) -> bool:
#         return self._tag == Tag.Empty
#
#     def is_throw(self) -> bool:
#         return self._tag == Tag.Throw
#
#     def as_throw(self) -> Optional[Exception]:
#         if self._tag == Tag.Throw:
#             return cast(Exception, self._union)
#         else:
#             return None
#
#     def is_resolve(self) -> bool:
#         return self._tag == Tag.Resolve
#
#     def as_resolve(self) -> Optional[T]:
#         if self._tag == Tag.Resolve:
#             return cast(Any, self._union)
#         else:
#             return None
#
#
# def box_is_non_empty(box: Box[Tagged[Any]]) -> bool:
#     return not box.value.is_empty()
#
#
# def box_apply_pred(box: Box[Tagged[Any]], pred: Callable[[Tagged[Any]], bool]) -> bool:
#     return pred(box.value)
#
#
# @dataclass(frozen=True, eq=False)
# class Future[T]:
#     _lb: LockBox[Tagged[T]]
#
#     @staticmethod
#     def empty() -> Future:
#         return Future(_lb=LockBox.new(Tagged.empty()))
#
#     @staticmethod
#     def pure(value: T) -> Future:
#         return Future(_lb=LockBox.new(Tagged.resolve(value)))
#
#     def throw(self, exc: Exception):
#         with self._lb as box:
#             if box.value.is_empty():
#                 box.value = Tagged.throw(exc)
#
#     def resolve(self, value: T):
#         with self._lb as box:
#             if box.value.is_empty():
#                 box.value = Tagged.resolve(value)
#
#     def attempt(self, fn: Callable[[], T]):
#         with self._lb as box:
#             if box.value.is_empty():
#                 try:
#                     value = fn()
#                     box.value = Tagged.resolve(value)
#                 except Exception as exc:
#                     if is_fatal_exception(exc):
#                         raise
#                     else:
#                         box.value = Tagged.throw(exc)
#
#     def is_empty(self) -> bool:
#         with self._lb as box:
#             return box.value.is_empty()
#
#     def raw_unwrap(self) -> Tagged[T]:
#         with self._lb as box:
#             return box.value
#
#     def unwrap(self) -> Optional[T]:
#         with self._lb as box:
#             tagged = box.value
#         if tagged._tag == Tag.Empty:
#             return None
#         if tagged._tag == Tag.Throw:
#             raise cast(Exception, tagged._union)
#         else:
#             return cast(Any, tagged._union)
#
#     def wait(self):
#         with self._lb as box:
#             self._lb._lock.wait_for(partial(box_is_non_empty, box))
#
#     def raw_wait(self, pred: Callable[[Tagged[T]], bool]):
#         with self._lb as box:
#             self._lb._lock.wait_for(partial(box_apply_pred, box, pred))
#
#
# @dataclass(frozen=False)
# class MutTagged[T]:
#     mutable: bool
#     tagged: Tagged[T]
#
#
# def box_is_non_empty_or_closed(box: Box[MutTagged[Any]]) -> bool:
#     return not box.value.mutable or not box.value.tagged.is_empty()
#
#
# def box_is_empty_or_closed(box: Box[MutTagged[Any]]) -> bool:
#     return not box.value.mutable or box.value.tagged.is_empty()
#
#
# def box_apply_pred_or_closed(box: Box[MutTagged[Any]], pred: Callable[[Tagged[Any]], bool]) -> bool:
#     return not box.value.mutable or pred(box.value.tagged)
#
#
# @dataclass(frozen=True, eq=False)
# class MutFuture[T]:
#     _lb: LockBox[MutTagged[T]]
#
#     @staticmethod
#     def empty() -> MutFuture:
#         return MutFuture(_lb=LockBox.new(MutTagged(True, Tagged.empty())))
#
#     @staticmethod
#     def pure(value: T) -> MutFuture:
#         return MutFuture(_lb=LockBox.new(MutTagged(True, Tagged.resolve(value))))
#
#     def throw(self, exc: Exception):
#         with self._lb as box:
#             if box.value.mutable:
#                 box.value.tagged = Tagged.throw(exc)
#
#     def resolve(self, value: T):
#         with self._lb as box:
#             if box.value.mutable:
#                 box.value.tagged = Tagged.resolve(value)
#
#     def attempt(self, fn: Callable[[], T]):
#         with self._lb as box:
#             if box.value.mutable:
#                 try:
#                     value = fn()
#                     box.value.tagged = Tagged.resolve(value)
#                 except Exception as exc:
#                     if is_fatal_exception(exc):
#                         raise
#                     else:
#                         box.value.tagged = Tagged.throw(exc)
#
#     def is_empty(self) -> bool:
#         with self._lb as box:
#             return box.value.tagged.is_empty()
#
#     def raw_unwrap(self) -> Tagged[T]:
#         with self._lb as box:
#             return box.value.tagged
#
#     def unwrap(self) -> Optional[T]:
#         with self._lb as box:
#             tagged = box.value.tagged
#         if tagged._tag == Tag.Empty:
#             return None
#         if tagged._tag == Tag.Throw:
#             raise cast(Exception, tagged._union)
#         else:
#             return cast(Any, tagged._union)
#
#     def clear(self):
#         with self._lb as box:
#             if box.value.mutable:
#                 box.value.tagged = Tagged.empty()
#
#     def close(self):
#         with self._lb as box:
#             box.value.mutable = False
#
#     def is_closed(self):
#         with self._lb as box:
#             return not box.value.mutable
#
#     def wait(self):
#         with self._lb as box:
#             self._lb._lock.wait_for(partial(box_is_non_empty_or_closed, box))
#
#     def wait_empty(self):
#         with self._lb as box:
#             self._lb._lock.wait_for(partial(box_is_empty_or_closed, box))
#
#     def raw_wait(self, pred: Callable[[Tagged[T]], bool]):
#         with self._lb as box:
#             self._lb._lock.wait_for(partial(box_apply_pred_or_closed, box, pred))
#
#
# ThreadId = NewType("ThreadId", int)
#
#
# @dataclass(frozen=True, eq=False)
# class Forker:
#     _executor: Executor
#     _parent: ThreadId
#     _start_exit: EventChain
#
#     def fork(self, name: str, task: Task) -> Optional[Tuple[ThreadId, EventChain]]:
#         return self._executor.fork(
#             parent=self._parent, start_exit=self._start_exit, name=name, task=task
#         )
#
#     def join_children(self) -> None:
#         self._executor.join_children(self._parent, start_exit=self._start_exit)
#
#
# @dataclass(frozen=True, eq=False)
# class TaskEnv:
#     logger: Logger
#     forker: Forker
#     start_exit: EventChain
#
#
# class Task(metaclass=ABCMeta):
#     @staticmethod
#     def callable(fn: Callable[[TaskEnv], None]) -> Task:
#         return CallableTask(_fn=fn)
#
#     @abstractmethod
#     def run(self, env: TaskEnv) -> None:
#         raise NotImplementedError
#
#     def cleanup(self, env: TaskEnv, exc: Optional[Exception]) -> None:
#         return
#
#     def handle(self, env: TaskEnv, child: int, exc: Optional[Exception]) -> None:
#         return
#
#
# @dataclass(frozen=True, eq=False)
# class CallableTask(Task):
#     _fn: Callable[[TaskEnv], None]
#
#     @override
#     def run(self, env: TaskEnv) -> None:
#         self._fn(env)
#
#
# @dataclass(frozen=True, eq=False)
# class ThreadState:
#     task: Task
#     start_exit: EventChain
#     thread: Thread
#     parent: ThreadId
#
#
# @dataclass(eq=False)
# class ExecutorState:
#     id_src: ThreadId
#     threads: Dict[ThreadId, ThreadState]
#     children: Dict[ThreadId, Set[ThreadId]]
#
#     @staticmethod
#     def empty() -> ExecutorState:
#         return ExecutorState(ThreadId(1), {}, {ThreadId(0): set()})
#
#
# @dataclass(frozen=True, eq=False)
# class Executor:
#     _logger: Logger
#     _lb: LockBox[ExecutorState]
#     _root_start_exit: EventChain
#
#     @staticmethod
#     def new(logger: Optional[Logger] = None) -> Executor:
#         if logger is None:
#             logging.basicConfig(level=logging.ERROR)
#             logger = logging.getLogger("Executor")
#         return Executor(
#             _logger=logger,
#             _lb=LockBox.new(ExecutorState.empty()),
#             _root_start_exit=EventChain.root(),
#         )
#
#     def forker(self):
#         return Forker(
#             _executor=self, _parent=ThreadId(0), _start_exit=self._root_start_exit
#         )
#
#     def _fork_run(
#         self,
#         parent: ThreadId,
#         name: str,
#         task: Task,
#         child: ThreadId,
#         uname: str,
#         child_start_exit: EventChain,
#     ) -> None:
#         logger = self._logger.getChild(uname)
#         forker = Forker(_executor=self, _parent=parent, _start_exit=child_start_exit)
#         env = TaskEnv(logger=logger, forker=forker, start_exit=child_start_exit)
#
#         exc: Optional[Exception] = None
#
#         self._logger.info('Running "%s" for %d as %d', name, parent, child)
#
#         try:
#             task.run(env)
#         except Exception as caught:
#             if is_fatal_exception(caught):
#                 self._root_start_exit.set()
#             exc = caught
#
#         self._logger.info('Stopping "%s" for %d as %d', name, parent, child)
#
#         self.join_children(child, child_start_exit)
#
#         self._logger.info('Cleaning up "%s" for %d as %d', name, parent, child)
#
#         try:
#             task.cleanup(env, exc)
#         except Exception as caught:
#             if is_fatal_exception(caught):
#                 self._root_start_exit.set()
#             exc = CleanupError(cleanup_error=caught, run_error=exc)
#
#         with self._lb as box:
#             del box.value.threads[child]
#             parent_children = box.value.children.get(parent)
#             if parent_children is not None:
#                 parent_children.remove(child)
#                 if not parent_children:
#                     del box.value.children[parent]
#             parent_state = box.value.threads.get(parent)
#
#         if parent_state is None:
#             if exc is not None:
#                 self._logger.info('Reraising "%s" for %d as %d', name, parent, child)
#                 raise exc
#         else:
#             self._logger.info('Handling "%s" for %d as %d', name, parent, child)
#             parent_state.task.handle(env, child, exc)
#
#     def fork(
#         self, parent: ThreadId, start_exit: EventChain, name: str, task: Task
#     ) -> Optional[Tuple[ThreadId, EventChain]]:
#         self._logger.info('Attempting fork "%s" for %d', name, parent)
#         if start_exit.is_set():
#             self._logger.info('Skipping fork "%s" for %d', name, parent)
#             return None
#         else:
#             child_start_exit = start_exit.child()
#             with self._lb as box:
#                 child = box.value.id_src
#                 uname = f"{name}_{child}"
#                 thread = Thread(
#                     name=uname,
#                     target=self._fork_run,
#                     args=(parent, name, task, child, uname, child_start_exit),
#                 )
#                 box.value.id_src = ThreadId(box.value.id_src + 1)
#                 thread_state = ThreadState(
#                     task=task, start_exit=start_exit, thread=thread, parent=parent
#                 )
#                 box.value.threads[child] = thread_state
#                 if parent in box.value.children:
#                     box.value.children[parent].add(child)
#                 else:
#                     box.value.children[parent] = set((child,))
#             self._logger.info('Starting fork "%s" for %d as %d', name, parent, child)
#             thread.start()
#             return (child, start_exit)
#
#     def join_children(self, parent: ThreadId, start_exit: EventChain):
#         self._logger.info("Joining children for %d", parent)
#         start_exit.set()
#         while True:
#             next_uids: List[ThreadId]
#             with self._lb as box:
#                 if parent in box.value.children:
#                     next_uids = list(reversed(box.value.children))
#                 else:
#                     next_uids = []
#             if not next_uids:
#                 break
#             else:
#                 for next_uid in next_uids:
#                     next_thread_state = box.value.threads[next_uid]
#                     self._logger.info(
#                         "Stopping %d for %d", next_uid, next_thread_state.parent
#                     )
#                     next_thread_state.start_exit.set()
#                 for next_uid in next_uids:
#                     next_thread_state = box.value.threads[next_uid]
#                     self._logger.info(
#                         "Joining %d for %d", next_uid, next_thread_state.parent
#                     )
#                     next_thread_state.thread.join()
#         self._logger.info("Joined children for %d", parent)
#
#     def join_all(self):
#         self._logger.info("Joining all")
#         self._root_start_exit.set()
#         while True:
#             next_uids: List[ThreadId]
#             with self._lb as box:
#                 if box.value.threads:
#                     next_uids = list(reversed(box.value.threads.keys()))
#                 else:
#                     next_uids = []
#             if not next_uids:
#                 break
#             else:
#                 for next_uid in next_uids:
#                     next_thread_state = box.value.threads[next_uid]
#                     self._logger.info(
#                         "Stopping %d for %d", next_uid, next_thread_state.parent
#                     )
#                     next_thread_state.start_exit.set()
#                 for next_uid in next_uids:
#                     next_thread_state = box.value.threads[next_uid]
#                     self._logger.info(
#                         "Joining %d for %d", next_uid, next_thread_state.parent
#                     )
#                     next_thread_state.thread.join()
#         self._logger.info("Joined all")
#
#
# @contextmanager
# def forking(logger: Optional[Logger] = None) -> Generator[Forker]:
#     executor = Executor.new(logger=logger)
#     forker = executor.forker()
#     try:
#         yield forker
#     finally:
#         executor.join_all()
#
