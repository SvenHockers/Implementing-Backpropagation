؃
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
Adam/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_14/bias/v
{
)Adam/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_14/kernel/v
�
+Adam/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_13/bias/v
{
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_13/kernel/v
�
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/v
�
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/v
�
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/v
{
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/v
�
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_9/kernel/v
�
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_14/bias/m
{
)Adam/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_14/kernel/m
�
+Adam/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_14/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_13/bias/m
{
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_13/kernel/m
�
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/m
�
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/m
�
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/m
{
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/m
�
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_9/kernel/m
�
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
:*
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
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:*
dtype0
�
conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
:*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
:*
dtype0
�
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
:*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:*
dtype0
�
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
�
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:*
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_517953

NoOpNoOp
�k
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�k
value�kB�k B�j
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

	optimizer

signatures*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
 trace_3* 
6
!trace_0
"trace_1
#trace_2
$trace_3* 
* 
�
%layer-0
&layer_with_weights-0
&layer-1
'layer-2
(layer_with_weights-1
(layer-3
)layer-4
*layer_with_weights-2
*layer-5
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
�
1layer_with_weights-0
1layer-0
2layer-1
3layer_with_weights-1
3layer-2
4layer-3
5layer_with_weights-2
5layer-4
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratem�m�m�m�m�m�m�m�m�m�m�m�v�v�v�v�v�v�v�v�v�v�v�v�*

Aserving_default* 
OI
VARIABLE_VALUEconv2d_9/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_9/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_10/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_10/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_11/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_11/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_12/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_12/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_13/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_13/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_14/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_14/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1*

B0*
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
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator* 
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias
 P_jit_compiled_convolution_op*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias
 ]_jit_compiled_convolution_op*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

kernel
bias
 j_jit_compiled_convolution_op*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
6
ttrace_0
utrace_1
vtrace_2
wtrace_3* 
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias
 ~_jit_compiled_convolution_op*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
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
<
�	variables
�	keras_api

�total

�count*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
.
%0
&1
'2
(3
)4
*5*
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

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
'
10
21
32
43
54*
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

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
rl
VARIABLE_VALUEAdam/conv2d_9/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_9/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_10/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_10/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_11/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_11/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_12/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_12/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_13/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_13/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_14/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_14/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_9/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_9/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_10/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_10/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_11/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_11/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_12/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_12/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_13/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_13/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_14/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_14/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp+Adam/conv2d_14/kernel/m/Read/ReadVariableOp)Adam/conv2d_14/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp+Adam/conv2d_14/kernel/v/Read/ReadVariableOp)Adam/conv2d_14/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_518676
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/conv2d_14/kernel/mAdam/conv2d_14/bias/mAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/conv2d_14/kernel/vAdam/conv2d_14/bias/v*7
Tin0
.2,*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_518815��
�
�
4__inference_smaller_autoencoder_layer_call_fn_517982
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517712�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
��
�
"__inference__traced_restore_518815
file_prefix:
 assignvariableop_conv2d_9_kernel:.
 assignvariableop_1_conv2d_9_bias:=
#assignvariableop_2_conv2d_10_kernel:/
!assignvariableop_3_conv2d_10_bias:=
#assignvariableop_4_conv2d_11_kernel:/
!assignvariableop_5_conv2d_11_bias:=
#assignvariableop_6_conv2d_12_kernel:/
!assignvariableop_7_conv2d_12_bias:=
#assignvariableop_8_conv2d_13_kernel:/
!assignvariableop_9_conv2d_13_bias:>
$assignvariableop_10_conv2d_14_kernel:0
"assignvariableop_11_conv2d_14_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: D
*assignvariableop_19_adam_conv2d_9_kernel_m:6
(assignvariableop_20_adam_conv2d_9_bias_m:E
+assignvariableop_21_adam_conv2d_10_kernel_m:7
)assignvariableop_22_adam_conv2d_10_bias_m:E
+assignvariableop_23_adam_conv2d_11_kernel_m:7
)assignvariableop_24_adam_conv2d_11_bias_m:E
+assignvariableop_25_adam_conv2d_12_kernel_m:7
)assignvariableop_26_adam_conv2d_12_bias_m:E
+assignvariableop_27_adam_conv2d_13_kernel_m:7
)assignvariableop_28_adam_conv2d_13_bias_m:E
+assignvariableop_29_adam_conv2d_14_kernel_m:7
)assignvariableop_30_adam_conv2d_14_bias_m:D
*assignvariableop_31_adam_conv2d_9_kernel_v:6
(assignvariableop_32_adam_conv2d_9_bias_v:E
+assignvariableop_33_adam_conv2d_10_kernel_v:7
)assignvariableop_34_adam_conv2d_10_bias_v:E
+assignvariableop_35_adam_conv2d_11_kernel_v:7
)assignvariableop_36_adam_conv2d_11_bias_v:E
+assignvariableop_37_adam_conv2d_12_kernel_v:7
)assignvariableop_38_adam_conv2d_12_bias_v:E
+assignvariableop_39_adam_conv2d_13_kernel_v:7
)assignvariableop_40_adam_conv2d_13_bias_v:E
+assignvariableop_41_adam_conv2d_14_kernel_v:7
)assignvariableop_42_adam_conv2d_14_bias_v:
identity_44��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_11_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_11_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_12_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_12_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_13_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_13_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_14_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_14_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_9_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_9_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_10_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_10_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_11_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_11_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_12_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_12_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_13_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_13_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_14_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_14_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_9_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_9_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_10_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_10_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_11_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_11_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_12_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_12_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_13_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_13_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_14_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_14_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
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
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_518487

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_2_layer_call_fn_518147

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�(
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_518292

inputsB
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:B
(conv2d_14_conv2d_readvariableop_resource:7
)conv2d_14_biasadd_readvariableop_resource:
identity�� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������f
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_12/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_13/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������f
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_13/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_14/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  r
conv2d_14/SigmoidSigmoidconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:���������  l
IdentityIdentityconv2d_14/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_smaller_autoencoder_layer_call_fn_517856
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517800�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_518524

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_2_layer_call_fn_517376
input_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517344w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517916
input_1-
sequential_2_517889:!
sequential_2_517891:-
sequential_2_517893:!
sequential_2_517895:-
sequential_2_517897:!
sequential_2_517899:-
sequential_3_517902:!
sequential_3_517904:-
sequential_3_517906:!
sequential_3_517908:-
sequential_3_517910:!
sequential_3_517912:
identity��$sequential_2/StatefulPartitionedCall�$sequential_3/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_517889sequential_2_517891sequential_2_517893sequential_2_517895sequential_2_517897sequential_2_517899*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517344�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_517902sequential_3_517904sequential_3_517906sequential_3_517908sequential_3_517910sequential_3_517912*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517604�
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
�
4__inference_smaller_autoencoder_layer_call_fn_518011
x!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517800�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�	
k
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518350

inputs
identity�;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������  *
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������  �
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������  a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������  W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_3_layer_call_fn_518375

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517180

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
g
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_518504

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_14_layer_call_fn_518513

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_3_layer_call_fn_518259

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517604�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
4__inference_smaller_autoencoder_layer_call_fn_517739
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517712�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�	
�
-__inference_sequential_2_layer_call_fn_518164

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517344w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
*__inference_conv2d_13_layer_call_fn_518476

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517236

inputs)
conv2d_9_517194:
conv2d_9_517196:*
conv2d_10_517212:
conv2d_10_517214:*
conv2d_11_517230:
conv2d_11_517232:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 gaussian_noise_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517180�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)gaussian_noise_1/PartitionedCall:output:0conv2d_9_517194conv2d_9_517196*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_10_517212conv2d_10_517214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_11_517230conv2d_11_517232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229�
IdentityIdentity*conv2d_11/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
*__inference_conv2d_11_layer_call_fn_518419

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_518430

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_9_layer_call_fn_518359

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�^
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518130
xN
4sequential_2_conv2d_9_conv2d_readvariableop_resource:C
5sequential_2_conv2d_9_biasadd_readvariableop_resource:O
5sequential_2_conv2d_10_conv2d_readvariableop_resource:D
6sequential_2_conv2d_10_biasadd_readvariableop_resource:O
5sequential_2_conv2d_11_conv2d_readvariableop_resource:D
6sequential_2_conv2d_11_biasadd_readvariableop_resource:O
5sequential_3_conv2d_12_conv2d_readvariableop_resource:D
6sequential_3_conv2d_12_biasadd_readvariableop_resource:O
5sequential_3_conv2d_13_conv2d_readvariableop_resource:D
6sequential_3_conv2d_13_biasadd_readvariableop_resource:O
5sequential_3_conv2d_14_conv2d_readvariableop_resource:D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:
identity��-sequential_2/conv2d_10/BiasAdd/ReadVariableOp�,sequential_2/conv2d_10/Conv2D/ReadVariableOp�-sequential_2/conv2d_11/BiasAdd/ReadVariableOp�,sequential_2/conv2d_11/Conv2D/ReadVariableOp�,sequential_2/conv2d_9/BiasAdd/ReadVariableOp�+sequential_2/conv2d_9/Conv2D/ReadVariableOp�-sequential_3/conv2d_12/BiasAdd/ReadVariableOp�,sequential_3/conv2d_12/Conv2D/ReadVariableOp�-sequential_3/conv2d_13/BiasAdd/ReadVariableOp�,sequential_3/conv2d_13/Conv2D/ReadVariableOp�-sequential_3/conv2d_14/BiasAdd/ReadVariableOp�,sequential_3/conv2d_14/Conv2D/ReadVariableOpT
#sequential_2/gaussian_noise_1/ShapeShapex*
T0*
_output_shapes
:u
0sequential_2/gaussian_noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    w
2sequential_2/gaussian_noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
@sequential_2/gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormal,sequential_2/gaussian_noise_1/Shape:output:0*
T0*/
_output_shapes
:���������  *
dtype0�
/sequential_2/gaussian_noise_1/random_normal/mulMulIsequential_2/gaussian_noise_1/random_normal/RandomStandardNormal:output:0;sequential_2/gaussian_noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:���������  �
+sequential_2/gaussian_noise_1/random_normalAddV23sequential_2/gaussian_noise_1/random_normal/mul:z:09sequential_2/gaussian_noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:���������  �
!sequential_2/gaussian_noise_1/addAddV2x/sequential_2/gaussian_noise_1/random_normal:z:0*
T0*/
_output_shapes
:���������  �
+sequential_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_9/Conv2DConv2D%sequential_2/gaussian_noise_1/add:z:03sequential_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
,sequential_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_9/BiasAddBiasAdd%sequential_2/conv2d_9/Conv2D:output:04sequential_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
sequential_2/conv2d_9/ReluRelu&sequential_2/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
$sequential_2/max_pooling2d_3/MaxPoolMaxPool(sequential_2/conv2d_9/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_10/Conv2DConv2D-sequential_2/max_pooling2d_3/MaxPool:output:04sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_10/ReluRelu'sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$sequential_2/max_pooling2d_4/MaxPoolMaxPool)sequential_2/conv2d_10/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
,sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_11/Conv2DConv2D-sequential_2/max_pooling2d_4/MaxPool:output:04sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_11/BiasAddBiasAdd&sequential_2/conv2d_11/Conv2D:output:05sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_11/ReluRelu'sequential_2/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:����������
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_12/Conv2DConv2D)sequential_2/conv2d_11/Relu:activations:04sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
"sequential_3/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      u
$sequential_3/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
 sequential_3/up_sampling2d_3/mulMul+sequential_3/up_sampling2d_3/Const:output:0-sequential_3/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:�
9sequential_3/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor)sequential_3/conv2d_12/Relu:activations:0$sequential_3/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_13/Conv2DConv2DJsequential_3/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
"sequential_3/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      u
$sequential_3/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
 sequential_3/up_sampling2d_4/mulMul+sequential_3/up_sampling2d_4/Const:output:0-sequential_3/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:�
9sequential_3/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor)sequential_3/conv2d_13/Relu:activations:0$sequential_3/up_sampling2d_4/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_14/Conv2DConv2DJsequential_3/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
sequential_3/conv2d_14/SigmoidSigmoid'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:���������  y
IdentityIdentity"sequential_3/conv2d_14/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp.^sequential_2/conv2d_10/BiasAdd/ReadVariableOp-^sequential_2/conv2d_10/Conv2D/ReadVariableOp.^sequential_2/conv2d_11/BiasAdd/ReadVariableOp-^sequential_2/conv2d_11/Conv2D/ReadVariableOp-^sequential_2/conv2d_9/BiasAdd/ReadVariableOp,^sequential_2/conv2d_9/Conv2D/ReadVariableOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2^
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp-sequential_2/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_10/Conv2D/ReadVariableOp,sequential_2/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_11/BiasAdd/ReadVariableOp-sequential_2/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_11/Conv2D/ReadVariableOp,sequential_2/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_9/BiasAdd/ReadVariableOp,sequential_2/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_9/Conv2D/ReadVariableOp+sequential_2/conv2d_9/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517886
input_1-
sequential_2_517859:!
sequential_2_517861:-
sequential_2_517863:!
sequential_2_517865:-
sequential_2_517867:!
sequential_2_517869:-
sequential_3_517872:!
sequential_3_517874:-
sequential_3_517876:!
sequential_3_517878:-
sequential_3_517880:!
sequential_3_517882:
identity��$sequential_2/StatefulPartitionedCall�$sequential_3/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_517859sequential_2_517861sequential_2_517863sequential_2_517865sequential_2_517867sequential_2_517869*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517236�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_517872sequential_3_517874sequential_3_517876sequential_3_517878sequential_3_517880sequential_3_517882*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517519�
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�	
�
-__inference_sequential_2_layer_call_fn_517251
input_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517712
x-
sequential_2_517685:!
sequential_2_517687:-
sequential_2_517689:!
sequential_2_517691:-
sequential_2_517693:!
sequential_2_517695:-
sequential_3_517698:!
sequential_3_517700:-
sequential_3_517702:!
sequential_3_517704:-
sequential_3_517706:!
sequential_3_517708:
identity��$sequential_2/StatefulPartitionedCall�$sequential_3/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallxsequential_2_517685sequential_2_517687sequential_2_517689sequential_2_517691sequential_2_517693sequential_2_517695*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517236�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_517698sequential_3_517700sequential_3_517702sequential_3_517704sequential_3_517706sequential_3_517708*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517519�
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_518410

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517800
x-
sequential_2_517773:!
sequential_2_517775:-
sequential_2_517777:!
sequential_2_517779:-
sequential_2_517781:!
sequential_2_517783:-
sequential_3_517786:!
sequential_3_517788:-
sequential_3_517790:!
sequential_3_517792:-
sequential_3_517794:!
sequential_3_517796:
identity��$sequential_2/StatefulPartitionedCall�$sequential_3/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallxsequential_2_517773sequential_2_517775sequential_2_517777sequential_2_517779sequential_2_517781sequential_2_517783*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_517344�
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_517786sequential_3_517788sequential_3_517790sequential_3_517792sequential_3_517794sequential_3_517796*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517604�
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:R N
/
_output_shapes
:���������  

_user_specified_namex
�!
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_518191

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  j
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
max_pooling2d_3/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_10/Conv2DConv2D max_pooling2d_3/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:����������
max_pooling2d_4/MaxPoolMaxPoolconv2d_10/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_11/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
IdentityIdentityconv2d_11/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_3_layer_call_fn_517534
conv2d_12_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517519�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_12_input
�
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517519

inputs*
conv2d_12_517477:
conv2d_12_517479:*
conv2d_13_517495:
conv2d_13_517497:*
conv2d_14_517513:
conv2d_14_517515:
identity��!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_517477conv2d_12_517479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476�
up_sampling2d_3/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_13_517495conv2d_13_517497*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494�
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_517513conv2d_14_517515*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512�
IdentityIdentity*conv2d_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_3_layer_call_fn_518242

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517519�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517398
input_2)
conv2d_9_517380:
conv2d_9_517382:*
conv2d_10_517386:
conv2d_10_517388:*
conv2d_11_517392:
conv2d_11_517394:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�
 gaussian_noise_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517180�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)gaussian_noise_1/PartitionedCall:output:0conv2d_9_517380conv2d_9_517382*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_10_517386conv2d_10_517388*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_11_517392conv2d_11_517394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229�
IdentityIdentity*conv2d_11/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
�
*__inference_conv2d_10_layer_call_fn_518389

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517420
input_2)
conv2d_9_517402:
conv2d_9_517404:*
conv2d_10_517408:
conv2d_10_517410:*
conv2d_11_517414:
conv2d_11_517416:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(gaussian_noise_1/StatefulPartitionedCall�
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517300�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0conv2d_9_517402conv2d_9_517404*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_10_517408conv2d_10_517410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_11_517414conv2d_11_517416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229�
IdentityIdentity*conv2d_11/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_2
�
g
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_518380

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�(
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_518325

inputsB
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource:7
)conv2d_13_biasadd_readvariableop_resource:B
(conv2d_14_conv2d_readvariableop_resource:7
)conv2d_14_biasadd_readvariableop_resource:
identity�� conv2d_12/BiasAdd/ReadVariableOp�conv2d_12/Conv2D/ReadVariableOp� conv2d_13/BiasAdd/ReadVariableOp�conv2d_13/Conv2D/ReadVariableOp� conv2d_14/BiasAdd/ReadVariableOp�conv2d_14/Conv2D/ReadVariableOp�
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������f
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_3/mulMulup_sampling2d_3/Const:output:0 up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_12/Relu:activations:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_13/Conv2DConv2D=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������f
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_13/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_14/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  r
conv2d_14/SigmoidSigmoidconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:���������  l
IdentityIdentityconv2d_14/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�l
�
!__inference__wrapped_model_517145
input_1b
Hsmaller_autoencoder_sequential_2_conv2d_9_conv2d_readvariableop_resource:W
Ismaller_autoencoder_sequential_2_conv2d_9_biasadd_readvariableop_resource:c
Ismaller_autoencoder_sequential_2_conv2d_10_conv2d_readvariableop_resource:X
Jsmaller_autoencoder_sequential_2_conv2d_10_biasadd_readvariableop_resource:c
Ismaller_autoencoder_sequential_2_conv2d_11_conv2d_readvariableop_resource:X
Jsmaller_autoencoder_sequential_2_conv2d_11_biasadd_readvariableop_resource:c
Ismaller_autoencoder_sequential_3_conv2d_12_conv2d_readvariableop_resource:X
Jsmaller_autoencoder_sequential_3_conv2d_12_biasadd_readvariableop_resource:c
Ismaller_autoencoder_sequential_3_conv2d_13_conv2d_readvariableop_resource:X
Jsmaller_autoencoder_sequential_3_conv2d_13_biasadd_readvariableop_resource:c
Ismaller_autoencoder_sequential_3_conv2d_14_conv2d_readvariableop_resource:X
Jsmaller_autoencoder_sequential_3_conv2d_14_biasadd_readvariableop_resource:
identity��Asmaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOp�@smaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOp�Asmaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOp�@smaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOp�@smaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOp�?smaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOp�Asmaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOp�@smaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOp�Asmaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOp�@smaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOp�Asmaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOp�@smaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOp�
?smaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOpHsmaller_autoencoder_sequential_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
0smaller_autoencoder/sequential_2/conv2d_9/Conv2DConv2Dinput_1Gsmaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
@smaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_2/conv2d_9/BiasAddBiasAdd9smaller_autoencoder/sequential_2/conv2d_9/Conv2D:output:0Hsmaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
.smaller_autoencoder/sequential_2/conv2d_9/ReluRelu:smaller_autoencoder/sequential_2/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
8smaller_autoencoder/sequential_2/max_pooling2d_3/MaxPoolMaxPool<smaller_autoencoder/sequential_2/conv2d_9/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
@smaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_2/conv2d_10/Conv2DConv2DAsmaller_autoencoder/sequential_2/max_pooling2d_3/MaxPool:output:0Hsmaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Asmaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOpJsmaller_autoencoder_sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2smaller_autoencoder/sequential_2/conv2d_10/BiasAddBiasAdd:smaller_autoencoder/sequential_2/conv2d_10/Conv2D:output:0Ismaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
/smaller_autoencoder/sequential_2/conv2d_10/ReluRelu;smaller_autoencoder/sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:����������
8smaller_autoencoder/sequential_2/max_pooling2d_4/MaxPoolMaxPool=smaller_autoencoder/sequential_2/conv2d_10/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
@smaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_2/conv2d_11/Conv2DConv2DAsmaller_autoencoder/sequential_2/max_pooling2d_4/MaxPool:output:0Hsmaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Asmaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOpJsmaller_autoencoder_sequential_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2smaller_autoencoder/sequential_2/conv2d_11/BiasAddBiasAdd:smaller_autoencoder/sequential_2/conv2d_11/Conv2D:output:0Ismaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
/smaller_autoencoder/sequential_2/conv2d_11/ReluRelu;smaller_autoencoder/sequential_2/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:����������
@smaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_3/conv2d_12/Conv2DConv2D=smaller_autoencoder/sequential_2/conv2d_11/Relu:activations:0Hsmaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Asmaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOpJsmaller_autoencoder_sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2smaller_autoencoder/sequential_3/conv2d_12/BiasAddBiasAdd:smaller_autoencoder/sequential_3/conv2d_12/Conv2D:output:0Ismaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
/smaller_autoencoder/sequential_3/conv2d_12/ReluRelu;smaller_autoencoder/sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:����������
6smaller_autoencoder/sequential_3/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
8smaller_autoencoder/sequential_3/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
4smaller_autoencoder/sequential_3/up_sampling2d_3/mulMul?smaller_autoencoder/sequential_3/up_sampling2d_3/Const:output:0Asmaller_autoencoder/sequential_3/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:�
Msmaller_autoencoder/sequential_3/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor=smaller_autoencoder/sequential_3/conv2d_12/Relu:activations:08smaller_autoencoder/sequential_3/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
@smaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_3/conv2d_13/Conv2DConv2D^smaller_autoencoder/sequential_3/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0Hsmaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
Asmaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOpJsmaller_autoencoder_sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2smaller_autoencoder/sequential_3/conv2d_13/BiasAddBiasAdd:smaller_autoencoder/sequential_3/conv2d_13/Conv2D:output:0Ismaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
/smaller_autoencoder/sequential_3/conv2d_13/ReluRelu;smaller_autoencoder/sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:����������
6smaller_autoencoder/sequential_3/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
8smaller_autoencoder/sequential_3/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
4smaller_autoencoder/sequential_3/up_sampling2d_4/mulMul?smaller_autoencoder/sequential_3/up_sampling2d_4/Const:output:0Asmaller_autoencoder/sequential_3/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:�
Msmaller_autoencoder/sequential_3/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor=smaller_autoencoder/sequential_3/conv2d_13/Relu:activations:08smaller_autoencoder/sequential_3/up_sampling2d_4/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
@smaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOpIsmaller_autoencoder_sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
1smaller_autoencoder/sequential_3/conv2d_14/Conv2DConv2D^smaller_autoencoder/sequential_3/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0Hsmaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
Asmaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpJsmaller_autoencoder_sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
2smaller_autoencoder/sequential_3/conv2d_14/BiasAddBiasAdd:smaller_autoencoder/sequential_3/conv2d_14/Conv2D:output:0Ismaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
2smaller_autoencoder/sequential_3/conv2d_14/SigmoidSigmoid;smaller_autoencoder/sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
IdentityIdentity6smaller_autoencoder/sequential_3/conv2d_14/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOpB^smaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOpA^smaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOpB^smaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOpA^smaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOpA^smaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOp@^smaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOpB^smaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOpA^smaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOpB^smaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOpA^smaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOpB^smaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOpA^smaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2�
Asmaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOpAsmaller_autoencoder/sequential_2/conv2d_10/BiasAdd/ReadVariableOp2�
@smaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOp@smaller_autoencoder/sequential_2/conv2d_10/Conv2D/ReadVariableOp2�
Asmaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOpAsmaller_autoencoder/sequential_2/conv2d_11/BiasAdd/ReadVariableOp2�
@smaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOp@smaller_autoencoder/sequential_2/conv2d_11/Conv2D/ReadVariableOp2�
@smaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOp@smaller_autoencoder/sequential_2/conv2d_9/BiasAdd/ReadVariableOp2�
?smaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOp?smaller_autoencoder/sequential_2/conv2d_9/Conv2D/ReadVariableOp2�
Asmaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOpAsmaller_autoencoder/sequential_3/conv2d_12/BiasAdd/ReadVariableOp2�
@smaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOp@smaller_autoencoder/sequential_3/conv2d_12/Conv2D/ReadVariableOp2�
Asmaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOpAsmaller_autoencoder/sequential_3/conv2d_13/BiasAdd/ReadVariableOp2�
@smaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOp@smaller_autoencoder/sequential_3/conv2d_13/Conv2D/ReadVariableOp2�
Asmaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOpAsmaller_autoencoder/sequential_3/conv2d_14/BiasAdd/ReadVariableOp2�
@smaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOp@smaller_autoencoder/sequential_3/conv2d_14/Conv2D/ReadVariableOp:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1
�
L
0__inference_up_sampling2d_4_layer_call_fn_518492

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
0__inference_up_sampling2d_3_layer_call_fn_518455

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_4_layer_call_fn_518405

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_518467

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_12_layer_call_fn_518439

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
M
1__inference_gaussian_noise_1_layer_call_fn_518330

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517180h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
h
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518339

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517604

inputs*
conv2d_12_517586:
conv2d_12_517588:*
conv2d_13_517592:
conv2d_13_517594:*
conv2d_14_517598:
conv2d_14_517600:
identity��!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_517586conv2d_12_517588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476�
up_sampling2d_3/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_13_517592conv2d_13_517594*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494�
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_517598conv2d_14_517600*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512�
IdentityIdentity*conv2d_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�U
�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518067
xN
4sequential_2_conv2d_9_conv2d_readvariableop_resource:C
5sequential_2_conv2d_9_biasadd_readvariableop_resource:O
5sequential_2_conv2d_10_conv2d_readvariableop_resource:D
6sequential_2_conv2d_10_biasadd_readvariableop_resource:O
5sequential_2_conv2d_11_conv2d_readvariableop_resource:D
6sequential_2_conv2d_11_biasadd_readvariableop_resource:O
5sequential_3_conv2d_12_conv2d_readvariableop_resource:D
6sequential_3_conv2d_12_biasadd_readvariableop_resource:O
5sequential_3_conv2d_13_conv2d_readvariableop_resource:D
6sequential_3_conv2d_13_biasadd_readvariableop_resource:O
5sequential_3_conv2d_14_conv2d_readvariableop_resource:D
6sequential_3_conv2d_14_biasadd_readvariableop_resource:
identity��-sequential_2/conv2d_10/BiasAdd/ReadVariableOp�,sequential_2/conv2d_10/Conv2D/ReadVariableOp�-sequential_2/conv2d_11/BiasAdd/ReadVariableOp�,sequential_2/conv2d_11/Conv2D/ReadVariableOp�,sequential_2/conv2d_9/BiasAdd/ReadVariableOp�+sequential_2/conv2d_9/Conv2D/ReadVariableOp�-sequential_3/conv2d_12/BiasAdd/ReadVariableOp�,sequential_3/conv2d_12/Conv2D/ReadVariableOp�-sequential_3/conv2d_13/BiasAdd/ReadVariableOp�,sequential_3/conv2d_13/Conv2D/ReadVariableOp�-sequential_3/conv2d_14/BiasAdd/ReadVariableOp�,sequential_3/conv2d_14/Conv2D/ReadVariableOp�
+sequential_2/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_9/Conv2DConv2Dx3sequential_2/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
,sequential_2/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_9/BiasAddBiasAdd%sequential_2/conv2d_9/Conv2D:output:04sequential_2/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
sequential_2/conv2d_9/ReluRelu&sequential_2/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
$sequential_2/max_pooling2d_3/MaxPoolMaxPool(sequential_2/conv2d_9/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_10/Conv2DConv2D-sequential_2/max_pooling2d_3/MaxPool:output:04sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_10/ReluRelu'sequential_2/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:����������
$sequential_2/max_pooling2d_4/MaxPoolMaxPool)sequential_2/conv2d_10/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
,sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_2/conv2d_11/Conv2DConv2D-sequential_2/max_pooling2d_4/MaxPool:output:04sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_2/conv2d_11/BiasAddBiasAdd&sequential_2/conv2d_11/Conv2D:output:05sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_2/conv2d_11/ReluRelu'sequential_2/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:����������
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_12/Conv2DConv2D)sequential_2/conv2d_11/Relu:activations:04sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_3/conv2d_12/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
"sequential_3/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      u
$sequential_3/up_sampling2d_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
 sequential_3/up_sampling2d_3/mulMul+sequential_3/up_sampling2d_3/Const:output:0-sequential_3/up_sampling2d_3/Const_1:output:0*
T0*
_output_shapes
:�
9sequential_3/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor)sequential_3/conv2d_12/Relu:activations:0$sequential_3/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������*
half_pixel_centers(�
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_13/Conv2DConv2DJsequential_3/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_3/conv2d_13/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
"sequential_3/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      u
$sequential_3/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
 sequential_3/up_sampling2d_4/mulMul+sequential_3/up_sampling2d_4/Const:output:0-sequential_3/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:�
9sequential_3/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor)sequential_3/conv2d_13/Relu:activations:0$sequential_3/up_sampling2d_4/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(�
,sequential_3/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_3/conv2d_14/Conv2DConv2DJsequential_3/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:04sequential_3/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
-sequential_3/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/conv2d_14/BiasAddBiasAdd&sequential_3/conv2d_14/Conv2D:output:05sequential_3/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
sequential_3/conv2d_14/SigmoidSigmoid'sequential_3/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:���������  y
IdentityIdentity"sequential_3/conv2d_14/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:���������  �
NoOpNoOp.^sequential_2/conv2d_10/BiasAdd/ReadVariableOp-^sequential_2/conv2d_10/Conv2D/ReadVariableOp.^sequential_2/conv2d_11/BiasAdd/ReadVariableOp-^sequential_2/conv2d_11/Conv2D/ReadVariableOp-^sequential_2/conv2d_9/BiasAdd/ReadVariableOp,^sequential_2/conv2d_9/Conv2D/ReadVariableOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp.^sequential_3/conv2d_14/BiasAdd/ReadVariableOp-^sequential_3/conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 2^
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp-sequential_2/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_10/Conv2D/ReadVariableOp,sequential_2/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_11/BiasAdd/ReadVariableOp-sequential_2/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_11/Conv2D/ReadVariableOp,sequential_2/conv2d_11/Conv2D/ReadVariableOp2\
,sequential_2/conv2d_9/BiasAdd/ReadVariableOp,sequential_2/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_2/conv2d_9/Conv2D/ReadVariableOp+sequential_2/conv2d_9/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_14/BiasAdd/ReadVariableOp-sequential_3/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_14/Conv2D/ReadVariableOp,sequential_3/conv2d_14/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_518370

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
j
1__inference_gaussian_noise_1_layer_call_fn_518335

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517300w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_518400

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517657
conv2d_12_input*
conv2d_12_517639:
conv2d_12_517641:*
conv2d_13_517645:
conv2d_13_517647:*
conv2d_14_517651:
conv2d_14_517653:
identity��!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_517639conv2d_12_517641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476�
up_sampling2d_3/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_13_517645conv2d_13_517647*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494�
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_517651conv2d_14_517653*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512�
IdentityIdentity*conv2d_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_12_input
�	
�
-__inference_sequential_3_layer_call_fn_517636
conv2d_12_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_517604�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_12_input
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517344

inputs)
conv2d_9_517326:
conv2d_9_517328:*
conv2d_10_517332:
conv2d_10_517334:*
conv2d_11_517338:
conv2d_11_517340:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�(gaussian_noise_1/StatefulPartitionedCall�
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517300�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0conv2d_9_517326conv2d_9_517328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_517193�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_517154�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_10_517332conv2d_10_517334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211�
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_517166�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_11_517338conv2d_11_517340*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_517229�
IdentityIdentity*conv2d_11/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_518450

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�)
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_518225

inputsA
'conv2d_9_conv2d_readvariableop_resource:6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOpL
gaussian_noise_1/ShapeShapeinputs*
T0*
_output_shapes
:h
#gaussian_noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%gaussian_noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
3gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_1/Shape:output:0*
T0*/
_output_shapes
:���������  *
dtype0�
"gaussian_noise_1/random_normal/mulMul<gaussian_noise_1/random_normal/RandomStandardNormal:output:0.gaussian_noise_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:���������  �
gaussian_noise_1/random_normalAddV2&gaussian_noise_1/random_normal/mul:z:0,gaussian_noise_1/random_normal/mean:output:0*
T0*/
_output_shapes
:���������  �
gaussian_noise_1/addAddV2inputs"gaussian_noise_1/random_normal:z:0*
T0*/
_output_shapes
:���������  �
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_9/Conv2DConv2Dgaussian_noise_1/add:z:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  j
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
max_pooling2d_3/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_10/Conv2DConv2D max_pooling2d_3/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:����������
max_pooling2d_4/MaxPoolMaxPoolconv2d_10/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_11/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:���������s
IdentityIdentityconv2d_11/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������  : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_517211

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�V
�
__inference__traced_save_518676
file_prefix.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_14_kernel_m_read_readvariableop4
0savev2_adam_conv2d_14_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop6
2savev2_adam_conv2d_14_kernel_v_read_readvariableop4
0savev2_adam_conv2d_14_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop2savev2_adam_conv2d_14_kernel_m_read_readvariableop0savev2_adam_conv2d_14_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop2savev2_adam_conv2d_14_kernel_v_read_readvariableop0savev2_adam_conv2d_14_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::::::: : : : : : : ::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,

_output_shapes
: 
�	
k
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_517300

inputs
identity�;
ShapeShapeinputs*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������  *
dtype0�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������  �
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������  a
addAddV2inputsrandom_normal:z:0*
T0*/
_output_shapes
:���������  W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  :W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517678
conv2d_12_input*
conv2d_12_517660:
conv2d_12_517662:*
conv2d_13_517666:
conv2d_13_517668:*
conv2d_14_517672:
conv2d_14_517674:
identity��!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_517660conv2d_12_517662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_517476�
up_sampling2d_3/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_517436�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_3/PartitionedCall:output:0conv2d_13_517666conv2d_13_517668*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_517494�
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_517455�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_14_517672conv2d_14_517674*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_517512�
IdentityIdentity*conv2d_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_12_input
�
�
$__inference_signature_wrapper_517953
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_517145w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_1"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������  D
output_18
StatefulPartitionedCall:0���������  tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
	decoder

	optimizer

signatures"
_tf_keras_model
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
 trace_32�
4__inference_smaller_autoencoder_layer_call_fn_517739
4__inference_smaller_autoencoder_layer_call_fn_517982
4__inference_smaller_autoencoder_layer_call_fn_518011
4__inference_smaller_autoencoder_layer_call_fn_517856�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2z trace_3
�
!trace_0
"trace_1
#trace_2
$trace_32�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518067
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518130
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517886
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517916�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z!trace_0z"trace_1z#trace_2z$trace_3
�B�
!__inference__wrapped_model_517145input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
%layer-0
&layer_with_weights-0
&layer-1
'layer-2
(layer_with_weights-1
(layer-3
)layer-4
*layer_with_weights-2
*layer-5
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
1layer_with_weights-0
1layer-0
2layer-1
3layer_with_weights-1
3layer-2
4layer-3
5layer_with_weights-2
5layer-4
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratem�m�m�m�m�m�m�m�m�m�m�m�v�v�v�v�v�v�v�v�v�v�v�v�"
	optimizer
,
Aserving_default"
signature_map
):'2conv2d_9/kernel
:2conv2d_9/bias
*:(2conv2d_10/kernel
:2conv2d_10/bias
*:(2conv2d_11/kernel
:2conv2d_11/bias
*:(2conv2d_12/kernel
:2conv2d_12/bias
*:(2conv2d_13/kernel
:2conv2d_13/bias
*:(2conv2d_14/kernel
:2conv2d_14/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_smaller_autoencoder_layer_call_fn_517739input_1"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
4__inference_smaller_autoencoder_layer_call_fn_517982x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
4__inference_smaller_autoencoder_layer_call_fn_518011x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
4__inference_smaller_autoencoder_layer_call_fn_517856input_1"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518067x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518130x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517886input_1"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517916input_1"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_random_generator"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias
 P_jit_compiled_convolution_op"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

kernel
bias
 ]_jit_compiled_convolution_op"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
�
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

kernel
bias
 j_jit_compiled_convolution_op"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_0
qtrace_1
rtrace_2
strace_32�
-__inference_sequential_2_layer_call_fn_517251
-__inference_sequential_2_layer_call_fn_518147
-__inference_sequential_2_layer_call_fn_518164
-__inference_sequential_2_layer_call_fn_517376�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0zqtrace_1zrtrace_2zstrace_3
�
ttrace_0
utrace_1
vtrace_2
wtrace_32�
H__inference_sequential_2_layer_call_and_return_conditional_losses_518191
H__inference_sequential_2_layer_call_and_return_conditional_losses_518225
H__inference_sequential_2_layer_call_and_return_conditional_losses_517398
H__inference_sequential_2_layer_call_and_return_conditional_losses_517420�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0zutrace_1zvtrace_2zwtrace_3
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_sequential_3_layer_call_fn_517534
-__inference_sequential_3_layer_call_fn_518242
-__inference_sequential_3_layer_call_fn_518259
-__inference_sequential_3_layer_call_fn_517636�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_sequential_3_layer_call_and_return_conditional_losses_518292
H__inference_sequential_3_layer_call_and_return_conditional_losses_518325
H__inference_sequential_3_layer_call_and_return_conditional_losses_517657
H__inference_sequential_3_layer_call_and_return_conditional_losses_517678�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_517953input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_gaussian_noise_1_layer_call_fn_518330
1__inference_gaussian_noise_1_layer_call_fn_518335�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518339
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518350�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_9_layer_call_fn_518359�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_518370�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_3_layer_call_fn_518375�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_518380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_10_layer_call_fn_518389�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_518400�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_4_layer_call_fn_518405�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_518410�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_11_layer_call_fn_518419�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_518430�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_2_layer_call_fn_517251input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_518147inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_518164inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_517376input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_518191inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_518225inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517398input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_517420input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_12_layer_call_fn_518439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_518450�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_up_sampling2d_3_layer_call_fn_518455�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_518467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_13_layer_call_fn_518476�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_518487�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_up_sampling2d_4_layer_call_fn_518492�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_518504�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_14_layer_call_fn_518513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_518524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
C
10
21
32
43
54"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_3_layer_call_fn_517534conv2d_12_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_3_layer_call_fn_518242inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_3_layer_call_fn_518259inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_3_layer_call_fn_517636conv2d_12_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_518292inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_518325inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517657conv2d_12_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_3_layer_call_and_return_conditional_losses_517678conv2d_12_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
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
�B�
1__inference_gaussian_noise_1_layer_call_fn_518330inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_gaussian_noise_1_layer_call_fn_518335inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518339inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518350inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
)__inference_conv2d_9_layer_call_fn_518359inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_9_layer_call_and_return_conditional_losses_518370inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_max_pooling2d_3_layer_call_fn_518375inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_518380inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_10_layer_call_fn_518389inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_518400inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_max_pooling2d_4_layer_call_fn_518405inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_518410inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_11_layer_call_fn_518419inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_518430inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_12_layer_call_fn_518439inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_518450inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_up_sampling2d_3_layer_call_fn_518455inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_518467inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_13_layer_call_fn_518476inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_518487inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_up_sampling2d_4_layer_call_fn_518492inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_518504inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_conv2d_14_layer_call_fn_518513inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_518524inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.:,2Adam/conv2d_9/kernel/m
 :2Adam/conv2d_9/bias/m
/:-2Adam/conv2d_10/kernel/m
!:2Adam/conv2d_10/bias/m
/:-2Adam/conv2d_11/kernel/m
!:2Adam/conv2d_11/bias/m
/:-2Adam/conv2d_12/kernel/m
!:2Adam/conv2d_12/bias/m
/:-2Adam/conv2d_13/kernel/m
!:2Adam/conv2d_13/bias/m
/:-2Adam/conv2d_14/kernel/m
!:2Adam/conv2d_14/bias/m
.:,2Adam/conv2d_9/kernel/v
 :2Adam/conv2d_9/bias/v
/:-2Adam/conv2d_10/kernel/v
!:2Adam/conv2d_10/bias/v
/:-2Adam/conv2d_11/kernel/v
!:2Adam/conv2d_11/bias/v
/:-2Adam/conv2d_12/kernel/v
!:2Adam/conv2d_12/bias/v
/:-2Adam/conv2d_13/kernel/v
!:2Adam/conv2d_13/bias/v
/:-2Adam/conv2d_14/kernel/v
!:2Adam/conv2d_14/bias/v�
!__inference__wrapped_model_517145�8�5
.�+
)�&
input_1���������  
� ";�8
6
output_1*�'
output_1���������  �
E__inference_conv2d_10_layer_call_and_return_conditional_losses_518400l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
*__inference_conv2d_10_layer_call_fn_518389_7�4
-�*
(�%
inputs���������
� " �����������
E__inference_conv2d_11_layer_call_and_return_conditional_losses_518430l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
*__inference_conv2d_11_layer_call_fn_518419_7�4
-�*
(�%
inputs���������
� " �����������
E__inference_conv2d_12_layer_call_and_return_conditional_losses_518450l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
*__inference_conv2d_12_layer_call_fn_518439_7�4
-�*
(�%
inputs���������
� " �����������
E__inference_conv2d_13_layer_call_and_return_conditional_losses_518487�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
*__inference_conv2d_13_layer_call_fn_518476�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
E__inference_conv2d_14_layer_call_and_return_conditional_losses_518524�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
*__inference_conv2d_14_layer_call_fn_518513�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
D__inference_conv2d_9_layer_call_and_return_conditional_losses_518370l7�4
-�*
(�%
inputs���������  
� "-�*
#� 
0���������  
� �
)__inference_conv2d_9_layer_call_fn_518359_7�4
-�*
(�%
inputs���������  
� " ����������  �
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518339l;�8
1�.
(�%
inputs���������  
p 
� "-�*
#� 
0���������  
� �
L__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_518350l;�8
1�.
(�%
inputs���������  
p
� "-�*
#� 
0���������  
� �
1__inference_gaussian_noise_1_layer_call_fn_518330_;�8
1�.
(�%
inputs���������  
p 
� " ����������  �
1__inference_gaussian_noise_1_layer_call_fn_518335_;�8
1�.
(�%
inputs���������  
p
� " ����������  �
K__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_518380�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_3_layer_call_fn_518375�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_518410�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_4_layer_call_fn_518405�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
H__inference_sequential_2_layer_call_and_return_conditional_losses_517398y@�=
6�3
)�&
input_2���������  
p 

 
� "-�*
#� 
0���������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_517420y@�=
6�3
)�&
input_2���������  
p

 
� "-�*
#� 
0���������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_518191x?�<
5�2
(�%
inputs���������  
p 

 
� "-�*
#� 
0���������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_518225x?�<
5�2
(�%
inputs���������  
p

 
� "-�*
#� 
0���������
� �
-__inference_sequential_2_layer_call_fn_517251l@�=
6�3
)�&
input_2���������  
p 

 
� " �����������
-__inference_sequential_2_layer_call_fn_517376l@�=
6�3
)�&
input_2���������  
p

 
� " �����������
-__inference_sequential_2_layer_call_fn_518147k?�<
5�2
(�%
inputs���������  
p 

 
� " �����������
-__inference_sequential_2_layer_call_fn_518164k?�<
5�2
(�%
inputs���������  
p

 
� " �����������
H__inference_sequential_3_layer_call_and_return_conditional_losses_517657�H�E
>�;
1�.
conv2d_12_input���������
p 

 
� "?�<
5�2
0+���������������������������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_517678�H�E
>�;
1�.
conv2d_12_input���������
p

 
� "?�<
5�2
0+���������������������������
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_518292x?�<
5�2
(�%
inputs���������
p 

 
� "-�*
#� 
0���������  
� �
H__inference_sequential_3_layer_call_and_return_conditional_losses_518325x?�<
5�2
(�%
inputs���������
p

 
� "-�*
#� 
0���������  
� �
-__inference_sequential_3_layer_call_fn_517534�H�E
>�;
1�.
conv2d_12_input���������
p 

 
� "2�/+����������������������������
-__inference_sequential_3_layer_call_fn_517636�H�E
>�;
1�.
conv2d_12_input���������
p

 
� "2�/+����������������������������
-__inference_sequential_3_layer_call_fn_518242}?�<
5�2
(�%
inputs���������
p 

 
� "2�/+����������������������������
-__inference_sequential_3_layer_call_fn_518259}?�<
5�2
(�%
inputs���������
p

 
� "2�/+����������������������������
$__inference_signature_wrapper_517953�C�@
� 
9�6
4
input_1)�&
input_1���������  ";�8
6
output_1*�'
output_1���������  �
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517886�H�E
.�+
)�&
input_1���������  
�

trainingp "?�<
5�2
0+���������������������������
� �
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_517916�H�E
.�+
)�&
input_1���������  
�

trainingp"?�<
5�2
0+���������������������������
� �
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518067�B�?
(�%
#� 
x���������  
�

trainingp "-�*
#� 
0���������  
� �
O__inference_smaller_autoencoder_layer_call_and_return_conditional_losses_518130�B�?
(�%
#� 
x���������  
�

trainingp"-�*
#� 
0���������  
� �
4__inference_smaller_autoencoder_layer_call_fn_517739�H�E
.�+
)�&
input_1���������  
�

trainingp "2�/+����������������������������
4__inference_smaller_autoencoder_layer_call_fn_517856�H�E
.�+
)�&
input_1���������  
�

trainingp"2�/+����������������������������
4__inference_smaller_autoencoder_layer_call_fn_517982�B�?
(�%
#� 
x���������  
�

trainingp "2�/+����������������������������
4__inference_smaller_autoencoder_layer_call_fn_518011�B�?
(�%
#� 
x���������  
�

trainingp"2�/+����������������������������
K__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_518467�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_up_sampling2d_3_layer_call_fn_518455�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_518504�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_up_sampling2d_4_layer_call_fn_518492�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������