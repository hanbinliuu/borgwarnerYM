# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: svm.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='svm.proto',
  package='',
  syntax='proto3',
  serialized_options=b'\n*com.yimuzn.bogehuana_detection.device.grpc',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\tsvm.proto\"6\n\x0cTrainRequest\x12\x11\n\tdata_path\x18\x01 \x01(\t\x12\x13\n\x0bsplit_ratio\x18\x02 \x01(\x02\"h\n\rTrainResponse\x12\x0e\n\x06status\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x11\n\tmodel_url\x18\x04 \x01(\t\x12\x11\n\tparameter\x18\x05 \x01(\t\"\"\n\rDetectRequest\x12\x11\n\timage_url\x18\x01 \x01(\t\"A\n\x0e\x44\x65tectResponse\x12\x0e\n\x06status\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x0e\n\x06result\x18\x03 \x01(\x05\x32;\n\x11SVMTrainingServer\x12&\n\x05train\x12\r.TrainRequest\x1a\x0e.TrainResponse2?\n\x12SVMDetectionServer\x12)\n\x06\x64\x65tect\x12\x0e.DetectRequest\x1a\x0f.DetectResponseB,\n*com.yimuzn.bogehuana_detection.device.grpcb\x06proto3'
)




_TRAINREQUEST = _descriptor.Descriptor(
  name='TrainRequest',
  full_name='TrainRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_path', full_name='TrainRequest.data_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='split_ratio', full_name='TrainRequest.split_ratio', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13,
  serialized_end=67,
)


_TRAINRESPONSE = _descriptor.Descriptor(
  name='TrainResponse',
  full_name='TrainResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='TrainResponse.status', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message', full_name='TrainResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_id', full_name='TrainResponse.model_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_url', full_name='TrainResponse.model_url', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='parameter', full_name='TrainResponse.parameter', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=69,
  serialized_end=173,
)


_DETECTREQUEST = _descriptor.Descriptor(
  name='DetectRequest',
  full_name='DetectRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_url', full_name='DetectRequest.image_url', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=175,
  serialized_end=209,
)


_DETECTRESPONSE = _descriptor.Descriptor(
  name='DetectResponse',
  full_name='DetectResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='DetectResponse.status', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message', full_name='DetectResponse.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='result', full_name='DetectResponse.result', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=211,
  serialized_end=276,
)

DESCRIPTOR.message_types_by_name['TrainRequest'] = _TRAINREQUEST
DESCRIPTOR.message_types_by_name['TrainResponse'] = _TRAINRESPONSE
DESCRIPTOR.message_types_by_name['DetectRequest'] = _DETECTREQUEST
DESCRIPTOR.message_types_by_name['DetectResponse'] = _DETECTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainRequest = _reflection.GeneratedProtocolMessageType('TrainRequest', (_message.Message,), {
  'DESCRIPTOR' : _TRAINREQUEST,
  '__module__' : 'svm_pb2'
  # @@protoc_insertion_point(class_scope:TrainRequest)
  })
_sym_db.RegisterMessage(TrainRequest)

TrainResponse = _reflection.GeneratedProtocolMessageType('TrainResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRAINRESPONSE,
  '__module__' : 'svm_pb2'
  # @@protoc_insertion_point(class_scope:TrainResponse)
  })
_sym_db.RegisterMessage(TrainResponse)

DetectRequest = _reflection.GeneratedProtocolMessageType('DetectRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETECTREQUEST,
  '__module__' : 'svm_pb2'
  # @@protoc_insertion_point(class_scope:DetectRequest)
  })
_sym_db.RegisterMessage(DetectRequest)

DetectResponse = _reflection.GeneratedProtocolMessageType('DetectResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTRESPONSE,
  '__module__' : 'svm_pb2'
  # @@protoc_insertion_point(class_scope:DetectResponse)
  })
_sym_db.RegisterMessage(DetectResponse)


DESCRIPTOR._options = None

_SVMTRAININGSERVER = _descriptor.ServiceDescriptor(
  name='SVMTrainingServer',
  full_name='SVMTrainingServer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=278,
  serialized_end=337,
  methods=[
  _descriptor.MethodDescriptor(
    name='train',
    full_name='SVMTrainingServer.train',
    index=0,
    containing_service=None,
    input_type=_TRAINREQUEST,
    output_type=_TRAINRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SVMTRAININGSERVER)

DESCRIPTOR.services_by_name['SVMTrainingServer'] = _SVMTRAININGSERVER


_SVMDETECTIONSERVER = _descriptor.ServiceDescriptor(
  name='SVMDetectionServer',
  full_name='SVMDetectionServer',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=339,
  serialized_end=402,
  methods=[
  _descriptor.MethodDescriptor(
    name='detect',
    full_name='SVMDetectionServer.detect',
    index=0,
    containing_service=None,
    input_type=_DETECTREQUEST,
    output_type=_DETECTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SVMDETECTIONSERVER)

DESCRIPTOR.services_by_name['SVMDetectionServer'] = _SVMDETECTIONSERVER

# @@protoc_insertion_point(module_scope)
