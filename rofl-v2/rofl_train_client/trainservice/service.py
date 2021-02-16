from trainservice import flservice_pb2
from trainservice import flservice_pb2_grpc
from concurrent import futures
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

import grpc

from analysis_wrapper import AnalysisClientWrapper

NUM_FLOATS_PER_BLOCK = 10000

client = AnalysisClientWrapper("../configs/example_config.yml")


class FLClientTrainService(flservice_pb2_grpc.FLClientTrainServiceServicer):
    def train_model(self, round_id, config, global_model_weights_list):
        client.set_weights(global_model_weights_list)
        update = client.train(round_id)
        return update

    def TrainForRound(self, request_iterator, context):
        config = next(request_iterator).config
        meta = next(request_iterator).meta_block_message
        result = []
        for (id, data) in enumerate(request_iterator):
            result += data.model_block.floats
            if id == meta.num_blocks - 1:
                break

        update = self.train_model(meta.round_id, config, result)

        num_blocks = len(update) // NUM_FLOATS_PER_BLOCK
        if  len(update) % NUM_FLOATS_PER_BLOCK != 0:
            num_blocks += 1

        meta_block_message = flservice_pb2.MetaFloatBlockMessage(model_id=config.model_id,
                                                                 round_id=meta.round_id,
                                                                 num_blocks=num_blocks,
                                                                 num_floats=len(update))
        yield flservice_pb2.ClientModelMessage(meta_block_message=meta_block_message)

        for block_id in range(num_blocks):
            begin = block_id * NUM_FLOATS_PER_BLOCK
            end =  (block_id + 1) * NUM_FLOATS_PER_BLOCK
            if end > len(update):
                end = len(update)
            float_block_message = flservice_pb2.FloatBlock(block_number=block_id, floats=update[begin:end])
            yield flservice_pb2.ClientModelMessage(model_block=float_block_message)


class DummyFLClientTrainService(FLClientTrainService):
    def train_model(self, round_id, config, global_model_weights_list):
        print("Train model for round %d" % round_id)
        return [0.001] * len(global_model_weights_list)


def serve(service, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    flservice_pb2_grpc.add_FLClientTrainServiceServicer_to_server(
        service, server)
    server.add_insecure_port("[::]:%d" % port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve(FLClientTrainService(), 50016)
    # serve(DummyFLClientTrainService(), 50016)