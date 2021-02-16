from trainservice import flservice_pb2
from trainservice import flservice_pb2_grpc

import grpc
import logging
import sys

from trainservice.analysis_wrapper.analysis_observer import AnalysisObserver

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class FLClientTrainObserver:
    def __init__(self, server_addr):
        channel = grpc.insecure_channel(server_addr)
        self.evaluator = AnalysisObserver("../configs/example_config.yml")
        self.stub = flservice_pb2_grpc.FlserviceStub(channel)
        logging.info("Ready for models")


    def handle_model(self, round_id, model_params):
        print("Received model %d" % round_id)
        score = self.evaluator.evaluate(model_params, round_id)
        print(score)
        logging.info(
            '[EVAL] Test (round,loss,accuracy): (%d, %f, %f)' %
            (round_id, score[0], score[1]))

    def observe_model_training(self, model_id):
        msg_iter = self.stub.ObserverModelTraining(flservice_pb2.ModelSelection(model_id=model_id))
        current_meta = None
        block_counter = 0
        buff = []
        for msg_matcher in msg_iter:
            if msg_matcher.HasField("params"):
                msg_matcher = msg_matcher.params
                if msg_matcher.HasField("config"):
                    continue
                elif msg_matcher.HasField("model_block"):
                    msg_matcher = msg_matcher.model_block
                    if msg_matcher.HasField("param_meta"):
                        current_meta = msg_matcher.param_meta
                        block_counter = 0
                        buff = []
                    elif msg_matcher.HasField("param_block"):
                        if not msg_matcher.param_block.block_number == block_counter:
                            raise Exception("Not aligned block")
                        buff.append(msg_matcher.param_block.data)
                        block_counter += 1
                        if block_counter == current_meta.num_blocks:
                            serialized = b''.join(buff)
                            params = flservice_pb2.FloatBlock()
                            params.ParseFromString(serialized)
                            self.handle_model(current_meta.round_id, params.floats)
            elif msg_matcher.HasField("error_message"):
                print("Error occurred: %s" % msg_matcher.error_message.msg)
                break
            elif msg_matcher.HasField("done_message"):
                print("Training is done")
                break


if __name__ == '__main__':
    observer = FLClientTrainObserver('localhost:50051')
    observer.observe_model_training(1)