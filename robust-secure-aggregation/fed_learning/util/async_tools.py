# if sys.platform == "linux" or sys.platform == "linux2":
#     ASYNC_MODE = 'gevent' # seems faster
#     # ASYNC_MODE = 'eventlet'
# else:
#     ASYNC_MODE = 'eventlet'

ASYNC_MODE = 'gevent'
if ASYNC_MODE == 'gevent':
    # print("Using gevent")
    from gevent import monkey
    monkey.patch_all()
    print("Monkey patching gevent")
    #from gevent.threadpool import ThreadPoolExecutor
elif ASYNC_MODE == 'eventlet':
    import eventlet
    eventlet.monkey_patch()
    print("Monkey patching")
    # from eventlet import tpool
    # _pool.submit = tpool.execute
elif ASYNC_MODE == 'threading':
    pass
else:
    raise TypeError

import os
import logging
import threading
import concurrent.futures
#from multiprocessing.pool import ThreadPool
import sys
import time

logger = logging.getLogger(__name__)

_pool = concurrent.futures.ThreadPoolExecutor()
#_pool = None

EVENTLET_THREADPOOL_SIZE = 1

def run_native(fn):
    def helper(*args, **kwargs):
        # logger.info('Calling fn "%s" from thread %s' % (fn.__name__, threading.get_ident()))
        # # print("Helloxxxxx")
        # fut = _pool.submit(fn, *args, **kwargs)
        # # print(f"Submitted {fut.done()}")
        # res = fut.result()
        # # print("Result received")
        # #
        # return res
        # logger.info('Scheduling fn "%s" on the same thread %s' % (fn.__name__, threading.get_ident()))
        logger.info('Scheduling fn "%s" on the same thread' % (fn.__name__))
        return fn(*args, **kwargs)
    return helper

def shutdown_pool():
    _pool.shutdown()
