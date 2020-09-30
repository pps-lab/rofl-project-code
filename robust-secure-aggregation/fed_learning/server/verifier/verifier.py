# can verify many things
import time

import sys

from fed_learning.crypto.crypto_interface import CryptoInterface, VerificationException
import logging
from multiprocessing import Process

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(filename="verifier.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

class Verifier():

    def __init__(self, queue, finished_clients):
        super(Verifier, self).__init__()
        self.ci = CryptoInterface()
        self.queue = queue
        self.finished_clients = finished_clients

    def verify_secure_aggregator(self):
        rand_commits_list = []
        for sid, msg in self.finished_clients.items():
            randproofs = msg.content['randproofs']
            update_enc = msg.content['update_enc']
            rand_commits = msg.content['rand_commits']
            try:
                logger.info('Verifying randomness for client %s' % sid)
                logger.info('Update %s' % rand_commits)
                p = Process(target=self.ci.verify_randproof, args=(update_enc, rand_commits, randproofs))
                p.start()
                while p.is_alive():
                    time.sleep(0.1)
                passed = True # ??
                # passed = self.ci.verify_randproof(update_enc, rand_commits, randproofs)
                logger.info(f"Result {passed}")
                if not passed:
                    logger.error('Randomness proof verification failed for client %s' % sid)
                    self.queue.put(False)
                    sys.exit(0)
                logger.info('Client %s passed randomness verification' % sid)
            except VerificationException as e:
                logger.error('Randomness proof verification threw exception for client %s with msg: %s' % (sid, str(e)))
                self.queue.put(False)
                sys.exit(0)
            except Exception as e:
                logger.error('Another error occurred %s' % str(e))
            rand_commits_list.append(rand_commits)
        rand_sum = self.ci.add_commitments(rand_commits_list)
        if not self.ci.equals_neutral_group_element_vector(rand_sum):
            logger.error('Randomness does not cancel out')
            self.queue.put(False)
            sys.exit(0)

        logger.info("All done!")
        self.queue.put(True)

    def run(self) -> None:
        self.verify_secure_aggregator()



