# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import fxgb_pb2 as fxgb__pb2


class FXGBWorkerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Init = channel.unary_unary(
        '/FXGBWorker/Init',
        request_serializer=fxgb__pb2.InitRequest.SerializeToString,
        response_deserializer=fxgb__pb2.WorkerResponse.FromString,
        )
    self.Train = channel.unary_unary(
        '/FXGBWorker/Train',
        request_serializer=fxgb__pb2.TrainRequest.SerializeToString,
        response_deserializer=fxgb__pb2.WorkerResponse.FromString,
        )


class FXGBWorkerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Init(self, request, context):
    """Initialize rabit and environment variables.
    When client receives this RPC, it can accept or reject the federated training session.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Train(self, request, context):
    """Load data and train
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FXGBWorkerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Init': grpc.unary_unary_rpc_method_handler(
          servicer.Init,
          request_deserializer=fxgb__pb2.InitRequest.FromString,
          response_serializer=fxgb__pb2.WorkerResponse.SerializeToString,
      ),
      'Train': grpc.unary_unary_rpc_method_handler(
          servicer.Train,
          request_deserializer=fxgb__pb2.TrainRequest.FromString,
          response_serializer=fxgb__pb2.WorkerResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FXGBWorker', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
