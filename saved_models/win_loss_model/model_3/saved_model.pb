Ô÷
Ô
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18¾

Adam/dense_143/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_143/bias/v
{
)Adam/dense_143/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_143/bias/v*
_output_shapes
:*
dtype0

Adam/dense_143/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_143/kernel/v

+Adam/dense_143/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_143/kernel/v*
_output_shapes

:
*
dtype0

Adam/dense_142/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_142/bias/v
{
)Adam/dense_142/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_142/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_142/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3
*(
shared_nameAdam/dense_142/kernel/v

+Adam/dense_142/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_142/kernel/v*
_output_shapes

:3
*
dtype0

Adam/dense_143/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_143/bias/m
{
)Adam/dense_143/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_143/bias/m*
_output_shapes
:*
dtype0

Adam/dense_143/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameAdam/dense_143/kernel/m

+Adam/dense_143/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_143/kernel/m*
_output_shapes

:
*
dtype0

Adam/dense_142/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_142/bias/m
{
)Adam/dense_142/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_142/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_142/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3
*(
shared_nameAdam/dense_142/kernel/m

+Adam/dense_142/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_142/kernel/m*
_output_shapes

:3
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
dense_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_143/bias
m
"dense_143/bias/Read/ReadVariableOpReadVariableOpdense_143/bias*
_output_shapes
:*
dtype0
|
dense_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_143/kernel
u
$dense_143/kernel/Read/ReadVariableOpReadVariableOpdense_143/kernel*
_output_shapes

:
*
dtype0
t
dense_142/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_142/bias
m
"dense_142/bias/Read/ReadVariableOpReadVariableOpdense_142/bias*
_output_shapes
:
*
dtype0
|
dense_142/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:3
*!
shared_namedense_142/kernel
u
$dense_142/kernel/Read/ReadVariableOpReadVariableOpdense_142/kernel*
_output_shapes

:3
*
dtype0

NoOpNoOp
é$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¤$
value$B$ B$
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
	
0* 
°
non_trainable_variables

layers
 metrics
!layer_regularization_losses
"layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
#trace_0
$trace_1
%trace_2
&trace_3* 
6
'trace_0
(trace_1
)trace_2
*trace_3* 
* 

+iter

,beta_1

-beta_2
	.decay
/learning_ratemKmLmMmNvOvPvQvR*

0serving_default* 

0
1*

0
1*
	
0* 

1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

6trace_0* 

7trace_0* 
`Z
VARIABLE_VALUEdense_142/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_142/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

=trace_0* 

>trace_0* 
`Z
VARIABLE_VALUEdense_143/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_143/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

?trace_0* 
* 

0
1
2*

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
B	variables
C	keras_api
	Dtotal
	Ecount*
H
F	variables
G	keras_api
	Htotal
	Icount
J
_fn_kwargs*

D0
E1*

B	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

F	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}
VARIABLE_VALUEAdam/dense_142/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_142/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_143/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_143/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_142/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_142/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_143/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_143/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_51Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ3

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_51dense_142/kerneldense_142/biasdense_143/kerneldense_143/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_14805725
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_142/kernel/Read/ReadVariableOp"dense_142/bias/Read/ReadVariableOp$dense_143/kernel/Read/ReadVariableOp"dense_143/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_142/kernel/m/Read/ReadVariableOp)Adam/dense_142/bias/m/Read/ReadVariableOp+Adam/dense_143/kernel/m/Read/ReadVariableOp)Adam/dense_143/bias/m/Read/ReadVariableOp+Adam/dense_142/kernel/v/Read/ReadVariableOp)Adam/dense_142/bias/v/Read/ReadVariableOp+Adam/dense_143/kernel/v/Read/ReadVariableOp)Adam/dense_143/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_14805948

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_142/kerneldense_142/biasdense_143/kerneldense_143/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_142/kernel/mAdam/dense_142/bias/mAdam/dense_143/kernel/mAdam/dense_143/bias/mAdam/dense_142/kernel/vAdam/dense_142/bias/vAdam/dense_143/kernel/vAdam/dense_143/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_14806021ë©
ø
Ë
&__inference_signature_wrapper_14805725
input_51
unknown:3

	unknown_0:

	unknown_1:

	unknown_2:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinput_51unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_14805514o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51
¼
ï
F__inference_model_50_layer_call_and_return_conditional_losses_14805634

inputs$
dense_142_14805617:3
 
dense_142_14805619:
$
dense_143_14805622:
 
dense_143_14805624:
identity¢!dense_142/StatefulPartitionedCall¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢!dense_143/StatefulPartitionedCallú
!dense_142/StatefulPartitionedCallStatefulPartitionedCallinputsdense_142_14805617dense_142_14805619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_14805622dense_143_14805624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_142_14805617*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp"^dense_142/StatefulPartitionedCall0^dense_142/kernel/Regularizer/Abs/ReadVariableOp"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
 
Ð
+__inference_model_50_layer_call_fn_14805658
input_51
unknown:3

	unknown_0:

	unknown_1:

	unknown_2:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_51unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_50_layer_call_and_return_conditional_losses_14805634o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51
¤U
¨
$__inference__traced_restore_14806021
file_prefix3
!assignvariableop_dense_142_kernel:3
/
!assignvariableop_1_dense_142_bias:
5
#assignvariableop_2_dense_143_kernel:
/
!assignvariableop_3_dense_143_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: #
assignvariableop_11_total: #
assignvariableop_12_count: =
+assignvariableop_13_adam_dense_142_kernel_m:3
7
)assignvariableop_14_adam_dense_142_bias_m:
=
+assignvariableop_15_adam_dense_143_kernel_m:
7
)assignvariableop_16_adam_dense_143_bias_m:=
+assignvariableop_17_adam_dense_142_kernel_v:3
7
)assignvariableop_18_adam_dense_142_bias_v:
=
+assignvariableop_19_adam_dense_143_kernel_v:
7
)assignvariableop_20_adam_dense_143_bias_v:
identity_22¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¾
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ä

valueÚ
B×
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_dense_142_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_142_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_143_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_143_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_dense_142_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_142_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_143_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_143_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_142_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_142_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_143_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_143_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Î
+__inference_model_50_layer_call_fn_14805744

inputs
unknown:3

	unknown_0:

	unknown_1:

	unknown_2:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_50_layer_call_and_return_conditional_losses_14805568o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
Â
ñ
F__inference_model_50_layer_call_and_return_conditional_losses_14805678
input_51$
dense_142_14805661:3
 
dense_142_14805663:
$
dense_143_14805666:
 
dense_143_14805668:
identity¢!dense_142/StatefulPartitionedCall¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢!dense_143/StatefulPartitionedCallü
!dense_142/StatefulPartitionedCallStatefulPartitionedCallinput_51dense_142_14805661dense_142_14805663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_14805666dense_143_14805668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_142_14805661*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp"^dense_142/StatefulPartitionedCall0^dense_142/kernel/Regularizer/Abs/ReadVariableOp"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51
 

#__inference__wrapped_model_14805514
input_51C
1model_50_dense_142_matmul_readvariableop_resource:3
@
2model_50_dense_142_biasadd_readvariableop_resource:
C
1model_50_dense_143_matmul_readvariableop_resource:
@
2model_50_dense_143_biasadd_readvariableop_resource:
identity¢)model_50/dense_142/BiasAdd/ReadVariableOp¢(model_50/dense_142/MatMul/ReadVariableOp¢)model_50/dense_143/BiasAdd/ReadVariableOp¢(model_50/dense_143/MatMul/ReadVariableOp
(model_50/dense_142/MatMul/ReadVariableOpReadVariableOp1model_50_dense_142_matmul_readvariableop_resource*
_output_shapes

:3
*
dtype0
model_50/dense_142/MatMulMatMulinput_510model_50/dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)model_50/dense_142/BiasAdd/ReadVariableOpReadVariableOp2model_50_dense_142_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¯
model_50/dense_142/BiasAddBiasAdd#model_50/dense_142/MatMul:product:01model_50/dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
v
model_50/dense_142/ReluRelu#model_50/dense_142/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(model_50/dense_143/MatMul/ReadVariableOpReadVariableOp1model_50_dense_143_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0®
model_50/dense_143/MatMulMatMul%model_50/dense_142/Relu:activations:00model_50/dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model_50/dense_143/BiasAdd/ReadVariableOpReadVariableOp2model_50_dense_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¯
model_50/dense_143/BiasAddBiasAdd#model_50/dense_143/MatMul:product:01model_50/dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
model_50/dense_143/SigmoidSigmoid#model_50/dense_143/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitymodel_50/dense_143/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
NoOpNoOp*^model_50/dense_142/BiasAdd/ReadVariableOp)^model_50/dense_142/MatMul/ReadVariableOp*^model_50/dense_143/BiasAdd/ReadVariableOp)^model_50/dense_143/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2V
)model_50/dense_142/BiasAdd/ReadVariableOp)model_50/dense_142/BiasAdd/ReadVariableOp2T
(model_50/dense_142/MatMul/ReadVariableOp(model_50/dense_142/MatMul/ReadVariableOp2V
)model_50/dense_143/BiasAdd/ReadVariableOp)model_50/dense_143/BiasAdd/ReadVariableOp2T
(model_50/dense_143/MatMul/ReadVariableOp(model_50/dense_143/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51
È

,__inference_dense_142_layer_call_fn_14805814

inputs
unknown:3

	unknown_0:

identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
Â
ñ
F__inference_model_50_layer_call_and_return_conditional_losses_14805698
input_51$
dense_142_14805681:3
 
dense_142_14805683:
$
dense_143_14805686:
 
dense_143_14805688:
identity¢!dense_142/StatefulPartitionedCall¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢!dense_143/StatefulPartitionedCallü
!dense_142/StatefulPartitionedCallStatefulPartitionedCallinput_51dense_142_14805681dense_142_14805683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_14805686dense_143_14805688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_142_14805681*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp"^dense_142/StatefulPartitionedCall0^dense_142/kernel/Regularizer/Abs/ReadVariableOp"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51


ø
G__inference_dense_143_layer_call_and_return_conditional_losses_14805851

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¼
ï
F__inference_model_50_layer_call_and_return_conditional_losses_14805568

inputs$
dense_142_14805539:3
 
dense_142_14805541:
$
dense_143_14805556:
 
dense_143_14805558:
identity¢!dense_142/StatefulPartitionedCall¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢!dense_143/StatefulPartitionedCallú
!dense_142/StatefulPartitionedCallStatefulPartitionedCallinputsdense_142_14805539dense_142_14805541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_14805556dense_143_14805558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_142_14805539*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_143/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp"^dense_142/StatefulPartitionedCall0^dense_142/kernel/Regularizer/Abs/ReadVariableOp"^dense_143/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
£
ª
G__inference_dense_142_layer_call_and_return_conditional_losses_14805831

inputs0
matmul_readvariableop_resource:3
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_142/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:3
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_142/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
¾

F__inference_model_50_layer_call_and_return_conditional_losses_14805805

inputs:
(dense_142_matmul_readvariableop_resource:3
7
)dense_142_biasadd_readvariableop_resource:
:
(dense_143_matmul_readvariableop_resource:
7
)dense_143_biasadd_readvariableop_resource:
identity¢ dense_142/BiasAdd/ReadVariableOp¢dense_142/MatMul/ReadVariableOp¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢ dense_143/BiasAdd/ReadVariableOp¢dense_143/MatMul/ReadVariableOp
dense_142/MatMul/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:3
*
dtype0}
dense_142/MatMulMatMulinputs'dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_142/BiasAdd/ReadVariableOpReadVariableOp)dense_142_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_142/BiasAddBiasAdddense_142/MatMul:product:0(dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_142/ReluReludense_142/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_143/MatMulMatMuldense_142/Relu:activations:0'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dense_143/SigmoidSigmoiddense_143/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentitydense_143/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_142/BiasAdd/ReadVariableOp ^dense_142/MatMul/ReadVariableOp0^dense_142/kernel/Regularizer/Abs/ReadVariableOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2D
 dense_142/BiasAdd/ReadVariableOp dense_142/BiasAdd/ReadVariableOp2B
dense_142/MatMul/ReadVariableOpdense_142/MatMul/ReadVariableOp2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
£
ª
G__inference_dense_142_layer_call_and_return_conditional_losses_14805538

inputs0
matmul_readvariableop_resource:3
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_142/kernel/Regularizer/Abs/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:3
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_142/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
¾

F__inference_model_50_layer_call_and_return_conditional_losses_14805781

inputs:
(dense_142_matmul_readvariableop_resource:3
7
)dense_142_biasadd_readvariableop_resource:
:
(dense_143_matmul_readvariableop_resource:
7
)dense_143_biasadd_readvariableop_resource:
identity¢ dense_142/BiasAdd/ReadVariableOp¢dense_142/MatMul/ReadVariableOp¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¢ dense_143/BiasAdd/ReadVariableOp¢dense_143/MatMul/ReadVariableOp
dense_142/MatMul/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:3
*
dtype0}
dense_142/MatMulMatMulinputs'dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 dense_142/BiasAdd/ReadVariableOpReadVariableOp)dense_142_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_142/BiasAddBiasAdddense_142/MatMul:product:0(dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
dense_142/ReluReludense_142/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_143/MatMulMatMuldense_142/Relu:activations:0'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dense_143/SigmoidSigmoiddense_143/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: d
IdentityIdentitydense_143/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_142/BiasAdd/ReadVariableOp ^dense_142/MatMul/ReadVariableOp0^dense_142/kernel/Regularizer/Abs/ReadVariableOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 2D
 dense_142/BiasAdd/ReadVariableOp dense_142/BiasAdd/ReadVariableOp2B
dense_142/MatMul/ReadVariableOpdense_142/MatMul/ReadVariableOp2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs

Î
+__inference_model_50_layer_call_fn_14805757

inputs
unknown:3

	unknown_0:

	unknown_1:

	unknown_2:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_50_layer_call_and_return_conditional_losses_14805634o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
 
_user_specified_nameinputs
 
Ð
+__inference_model_50_layer_call_fn_14805579
input_51
unknown:3

	unknown_0:

	unknown_1:

	unknown_2:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinput_51unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_model_50_layer_call_and_return_conditional_losses_14805568o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ3: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3
"
_user_specified_name
input_51


ø
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ª
¯
__inference_loss_fn_0_14805862J
8dense_142_kernel_regularizer_abs_readvariableop_resource:3

identity¢/dense_142/kernel/Regularizer/Abs/ReadVariableOp¨
/dense_142/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8dense_142_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:3
*
dtype0
 dense_142/kernel/Regularizer/AbsAbs7dense_142/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:3
s
"dense_142/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_142/kernel/Regularizer/SumSum$dense_142/kernel/Regularizer/Abs:y:0+dense_142/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_142/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 dense_142/kernel/Regularizer/mulMul+dense_142/kernel/Regularizer/mul/x:output:0)dense_142/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_142/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^dense_142/kernel/Regularizer/Abs/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/dense_142/kernel/Regularizer/Abs/ReadVariableOp/dense_142/kernel/Regularizer/Abs/ReadVariableOp
È

,__inference_dense_143_layer_call_fn_14805840

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_143_layer_call_and_return_conditional_losses_14805555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ê1
â
!__inference__traced_save_14805948
file_prefix/
+savev2_dense_142_kernel_read_readvariableop-
)savev2_dense_142_bias_read_readvariableop/
+savev2_dense_143_kernel_read_readvariableop-
)savev2_dense_143_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_142_kernel_m_read_readvariableop4
0savev2_adam_dense_142_bias_m_read_readvariableop6
2savev2_adam_dense_143_kernel_m_read_readvariableop4
0savev2_adam_dense_143_bias_m_read_readvariableop6
2savev2_adam_dense_142_kernel_v_read_readvariableop4
0savev2_adam_dense_142_bias_v_read_readvariableop6
2savev2_adam_dense_143_kernel_v_read_readvariableop4
0savev2_adam_dense_143_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: »
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ä

valueÚ
B×
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B å
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_142_kernel_read_readvariableop)savev2_dense_142_bias_read_readvariableop+savev2_dense_143_kernel_read_readvariableop)savev2_dense_143_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_142_kernel_m_read_readvariableop0savev2_adam_dense_142_bias_m_read_readvariableop2savev2_adam_dense_143_kernel_m_read_readvariableop0savev2_adam_dense_143_bias_m_read_readvariableop2savev2_adam_dense_142_kernel_v_read_readvariableop0savev2_adam_dense_142_bias_v_read_readvariableop2savev2_adam_dense_143_kernel_v_read_readvariableop0savev2_adam_dense_143_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesx
v: :3
:
:
:: : : : : : : : : :3
:
:
::3
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:3
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:3
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:3
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: "¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
=
input_511
serving_default_input_51:0ÿÿÿÿÿÿÿÿÿ3=
	dense_1430
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:©_
¾
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
0"
trackable_list_wrapper
Ê
non_trainable_variables

layers
 metrics
!layer_regularization_losses
"layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
â
#trace_0
$trace_1
%trace_2
&trace_32÷
+__inference_model_50_layer_call_fn_14805579
+__inference_model_50_layer_call_fn_14805744
+__inference_model_50_layer_call_fn_14805757
+__inference_model_50_layer_call_fn_14805658À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z#trace_0z$trace_1z%trace_2z&trace_3
Î
'trace_0
(trace_1
)trace_2
*trace_32ã
F__inference_model_50_layer_call_and_return_conditional_losses_14805781
F__inference_model_50_layer_call_and_return_conditional_losses_14805805
F__inference_model_50_layer_call_and_return_conditional_losses_14805678
F__inference_model_50_layer_call_and_return_conditional_losses_14805698À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z'trace_0z(trace_1z)trace_2z*trace_3
ÏBÌ
#__inference__wrapped_model_14805514input_51"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

+iter

,beta_1

-beta_2
	.decay
/learning_ratemKmLmMmNvOvPvQvR"
	optimizer
,
0serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
6trace_02Ó
,__inference_dense_142_layer_call_fn_14805814¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z6trace_0

7trace_02î
G__inference_dense_142_layer_call_and_return_conditional_losses_14805831¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z7trace_0
": 3
2dense_142/kernel
:
2dense_142/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ð
=trace_02Ó
,__inference_dense_143_layer_call_fn_14805840¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z=trace_0

>trace_02î
G__inference_dense_143_layer_call_and_return_conditional_losses_14805851¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z>trace_0
": 
2dense_143/kernel
:2dense_143/bias
Ï
?trace_02²
__inference_loss_fn_0_14805862
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z?trace_0
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
+__inference_model_50_layer_call_fn_14805579input_51"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ýBú
+__inference_model_50_layer_call_fn_14805744inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ýBú
+__inference_model_50_layer_call_fn_14805757inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÿBü
+__inference_model_50_layer_call_fn_14805658input_51"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_model_50_layer_call_and_return_conditional_losses_14805781inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_model_50_layer_call_and_return_conditional_losses_14805805inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_model_50_layer_call_and_return_conditional_losses_14805678input_51"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_model_50_layer_call_and_return_conditional_losses_14805698input_51"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÎBË
&__inference_signature_wrapper_14805725input_51"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_dense_142_layer_call_fn_14805814inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_dense_142_layer_call_and_return_conditional_losses_14805831inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_dense_143_layer_call_fn_14805840inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_dense_143_layer_call_and_return_conditional_losses_14805851inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µB²
__inference_loss_fn_0_14805862"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
N
B	variables
C	keras_api
	Dtotal
	Ecount"
_tf_keras_metric
^
F	variables
G	keras_api
	Htotal
	Icount
J
_fn_kwargs"
_tf_keras_metric
.
D0
E1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
:  (2total
:  (2count
.
H0
I1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%3
2Adam/dense_142/kernel/m
!:
2Adam/dense_142/bias/m
':%
2Adam/dense_143/kernel/m
!:2Adam/dense_143/bias/m
':%3
2Adam/dense_142/kernel/v
!:
2Adam/dense_142/bias/v
':%
2Adam/dense_143/kernel/v
!:2Adam/dense_143/bias/v
#__inference__wrapped_model_14805514p1¢.
'¢$
"
input_51ÿÿÿÿÿÿÿÿÿ3
ª "5ª2
0
	dense_143# 
	dense_143ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_142_layer_call_and_return_conditional_losses_14805831\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ3
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
,__inference_dense_142_layer_call_fn_14805814O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ3
ª "ÿÿÿÿÿÿÿÿÿ
§
G__inference_dense_143_layer_call_and_return_conditional_losses_14805851\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_143_layer_call_fn_14805840O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ=
__inference_loss_fn_0_14805862¢

¢ 
ª " ²
F__inference_model_50_layer_call_and_return_conditional_losses_14805678h9¢6
/¢,
"
input_51ÿÿÿÿÿÿÿÿÿ3
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
F__inference_model_50_layer_call_and_return_conditional_losses_14805698h9¢6
/¢,
"
input_51ÿÿÿÿÿÿÿÿÿ3
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_model_50_layer_call_and_return_conditional_losses_14805781f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ3
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_model_50_layer_call_and_return_conditional_losses_14805805f7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ3
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_model_50_layer_call_fn_14805579[9¢6
/¢,
"
input_51ÿÿÿÿÿÿÿÿÿ3
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_50_layer_call_fn_14805658[9¢6
/¢,
"
input_51ÿÿÿÿÿÿÿÿÿ3
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_50_layer_call_fn_14805744Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ3
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_model_50_layer_call_fn_14805757Y7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ3
p

 
ª "ÿÿÿÿÿÿÿÿÿ¦
&__inference_signature_wrapper_14805725|=¢:
¢ 
3ª0
.
input_51"
input_51ÿÿÿÿÿÿÿÿÿ3"5ª2
0
	dense_143# 
	dense_143ÿÿÿÿÿÿÿÿÿ