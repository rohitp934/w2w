??;
?&?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
q
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
?
StringFormat
inputs2T

output"
T
list(type)("
templatestring%s"
placeholderstring%s"
	summarizeint
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*2.9.12v2.9.0-18-gd8ce9f9c3018ȧ4
?
>Adagrad/multi_head_attention/attention_output/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*O
shared_name@>Adagrad/multi_head_attention/attention_output/bias/accumulator
?
RAdagrad/multi_head_attention/attention_output/bias/accumulator/Read/ReadVariableOpReadVariableOp>Adagrad/multi_head_attention/attention_output/bias/accumulator*
_output_shapes
:>*
dtype0
?
@Adagrad/multi_head_attention/attention_output/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*Q
shared_nameB@Adagrad/multi_head_attention/attention_output/kernel/accumulator
?
TAdagrad/multi_head_attention/attention_output/kernel/accumulator/Read/ReadVariableOpReadVariableOp@Adagrad/multi_head_attention/attention_output/kernel/accumulator*"
_output_shapes
:>>*
dtype0
?
3Adagrad/multi_head_attention/value/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*D
shared_name53Adagrad/multi_head_attention/value/bias/accumulator
?
GAdagrad/multi_head_attention/value/bias/accumulator/Read/ReadVariableOpReadVariableOp3Adagrad/multi_head_attention/value/bias/accumulator*
_output_shapes

:>*
dtype0
?
5Adagrad/multi_head_attention/value/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*F
shared_name75Adagrad/multi_head_attention/value/kernel/accumulator
?
IAdagrad/multi_head_attention/value/kernel/accumulator/Read/ReadVariableOpReadVariableOp5Adagrad/multi_head_attention/value/kernel/accumulator*"
_output_shapes
:>>*
dtype0
?
1Adagrad/multi_head_attention/key/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*B
shared_name31Adagrad/multi_head_attention/key/bias/accumulator
?
EAdagrad/multi_head_attention/key/bias/accumulator/Read/ReadVariableOpReadVariableOp1Adagrad/multi_head_attention/key/bias/accumulator*
_output_shapes

:>*
dtype0
?
3Adagrad/multi_head_attention/key/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*D
shared_name53Adagrad/multi_head_attention/key/kernel/accumulator
?
GAdagrad/multi_head_attention/key/kernel/accumulator/Read/ReadVariableOpReadVariableOp3Adagrad/multi_head_attention/key/kernel/accumulator*"
_output_shapes
:>>*
dtype0
?
3Adagrad/multi_head_attention/query/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*D
shared_name53Adagrad/multi_head_attention/query/bias/accumulator
?
GAdagrad/multi_head_attention/query/bias/accumulator/Read/ReadVariableOpReadVariableOp3Adagrad/multi_head_attention/query/bias/accumulator*
_output_shapes

:>*
dtype0
?
5Adagrad/multi_head_attention/query/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*F
shared_name75Adagrad/multi_head_attention/query/kernel/accumulator
?
IAdagrad/multi_head_attention/query/kernel/accumulator/Read/ReadVariableOpReadVariableOp5Adagrad/multi_head_attention/query/kernel/accumulator*"
_output_shapes
:>>*
dtype0
?
 Adagrad/dense_3/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adagrad/dense_3/bias/accumulator
?
4Adagrad/dense_3/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_3/bias/accumulator*
_output_shapes
:*
dtype0
?
"Adagrad/dense_3/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adagrad/dense_3/kernel/accumulator
?
6Adagrad/dense_3/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_3/kernel/accumulator*
_output_shapes
:	?*
dtype0
?
.Adagrad/batch_normalization_1/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*?
shared_name0.Adagrad/batch_normalization_1/beta/accumulator
?
BAdagrad/batch_normalization_1/beta/accumulator/Read/ReadVariableOpReadVariableOp.Adagrad/batch_normalization_1/beta/accumulator*
_output_shapes	
:?*
dtype0
?
/Adagrad/batch_normalization_1/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*@
shared_name1/Adagrad/batch_normalization_1/gamma/accumulator
?
CAdagrad/batch_normalization_1/gamma/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/batch_normalization_1/gamma/accumulator*
_output_shapes	
:?*
dtype0
?
 Adagrad/dense_2/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adagrad/dense_2/bias/accumulator
?
4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_2/bias/accumulator*
_output_shapes	
:?*
dtype0
?
"Adagrad/dense_2/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adagrad/dense_2/kernel/accumulator
?
6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_2/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
,Adagrad/batch_normalization/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adagrad/batch_normalization/beta/accumulator
?
@Adagrad/batch_normalization/beta/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/batch_normalization/beta/accumulator*
_output_shapes	
:?*
dtype0
?
-Adagrad/batch_normalization/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*>
shared_name/-Adagrad/batch_normalization/gamma/accumulator
?
AAdagrad/batch_normalization/gamma/accumulator/Read/ReadVariableOpReadVariableOp-Adagrad/batch_normalization/gamma/accumulator*
_output_shapes	
:?*
dtype0
?
 Adagrad/dense_1/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adagrad/dense_1/bias/accumulator
?
4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense_1/bias/accumulator*
_output_shapes	
:?*
dtype0
?
"Adagrad/dense_1/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adagrad/dense_1/kernel/accumulator
?
6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOpReadVariableOp"Adagrad/dense_1/kernel/accumulator* 
_output_shapes
:
??*
dtype0
?
.Adagrad/layer_normalization_1/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*?
shared_name0.Adagrad/layer_normalization_1/beta/accumulator
?
BAdagrad/layer_normalization_1/beta/accumulator/Read/ReadVariableOpReadVariableOp.Adagrad/layer_normalization_1/beta/accumulator*
_output_shapes
:>*
dtype0
?
/Adagrad/layer_normalization_1/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*@
shared_name1/Adagrad/layer_normalization_1/gamma/accumulator
?
CAdagrad/layer_normalization_1/gamma/accumulator/Read/ReadVariableOpReadVariableOp/Adagrad/layer_normalization_1/gamma/accumulator*
_output_shapes
:>*
dtype0
?
3Adagrad/occupation_embedding/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53Adagrad/occupation_embedding/embeddings/accumulator
?
GAdagrad/occupation_embedding/embeddings/accumulator/Read/ReadVariableOpReadVariableOp3Adagrad/occupation_embedding/embeddings/accumulator*
_output_shapes

:*
dtype0
?
2Adagrad/age_group_embedding/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42Adagrad/age_group_embedding/embeddings/accumulator
?
FAdagrad/age_group_embedding/embeddings/accumulator/Read/ReadVariableOpReadVariableOp2Adagrad/age_group_embedding/embeddings/accumulator*
_output_shapes

:*
dtype0
?
,Adagrad/sex_embedding/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adagrad/sex_embedding/embeddings/accumulator
?
@Adagrad/sex_embedding/embeddings/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/sex_embedding/embeddings/accumulator*
_output_shapes

:*
dtype0
?
Adagrad/dense/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*/
shared_name Adagrad/dense/bias/accumulator
?
2Adagrad/dense/bias/accumulator/Read/ReadVariableOpReadVariableOpAdagrad/dense/bias/accumulator*
_output_shapes
:>*
dtype0
?
 Adagrad/dense/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*1
shared_name" Adagrad/dense/kernel/accumulator
?
4Adagrad/dense/kernel/accumulator/Read/ReadVariableOpReadVariableOp Adagrad/dense/kernel/accumulator*
_output_shapes

:>>*
dtype0
?
,Adagrad/layer_normalization/beta/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*=
shared_name.,Adagrad/layer_normalization/beta/accumulator
?
@Adagrad/layer_normalization/beta/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/layer_normalization/beta/accumulator*
_output_shapes
:>*
dtype0
?
-Adagrad/layer_normalization/gamma/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*>
shared_name/-Adagrad/layer_normalization/gamma/accumulator
?
AAdagrad/layer_normalization/gamma/accumulator/Read/ReadVariableOpReadVariableOp-Adagrad/layer_normalization/gamma/accumulator*
_output_shapes
:>*
dtype0
?
<Adagrad/process_movie_embedding_with_genres/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*M
shared_name><Adagrad/process_movie_embedding_with_genres/bias/accumulator
?
PAdagrad/process_movie_embedding_with_genres/bias/accumulator/Read/ReadVariableOpReadVariableOp<Adagrad/process_movie_embedding_with_genres/bias/accumulator*
_output_shapes
:>*
dtype0
?
>Adagrad/process_movie_embedding_with_genres/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P>*O
shared_name@>Adagrad/process_movie_embedding_with_genres/kernel/accumulator
?
RAdagrad/process_movie_embedding_with_genres/kernel/accumulator/Read/ReadVariableOpReadVariableOp>Adagrad/process_movie_embedding_with_genres/kernel/accumulator*
_output_shapes

:P>*
dtype0
?
.Adagrad/movie_embedding/embeddings/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?>*?
shared_name0.Adagrad/movie_embedding/embeddings/accumulator
?
BAdagrad/movie_embedding/embeddings/accumulator/Read/ReadVariableOpReadVariableOp.Adagrad/movie_embedding/embeddings/accumulator*
_output_shapes
:	?>*
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
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
?
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*;
shared_name,*multi_head_attention/attention_output/bias
?
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:>*
dtype0
?
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*=
shared_name.,multi_head_attention/attention_output/kernel
?
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:>>*
dtype0
?
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*0
shared_name!multi_head_attention/value/bias
?
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:>*
dtype0
?
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*2
shared_name#!multi_head_attention/value/kernel
?
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:>>*
dtype0
?
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*.
shared_namemulti_head_attention/key/bias
?
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:>*
dtype0
?
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*0
shared_name!multi_head_attention/key/kernel
?
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:>>*
dtype0
?
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*0
shared_name!multi_head_attention/query/bias
?
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:>*
dtype0
?
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:>>*2
shared_name#!multi_head_attention/query/kernel
?
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:>>*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
?
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_namelayer_normalization_1/beta
?
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:>*
dtype0
?
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*,
shared_namelayer_normalization_1/gamma
?
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:>*
dtype0
?
occupation_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!occupation_embedding/embeddings
?
3occupation_embedding/embeddings/Read/ReadVariableOpReadVariableOpoccupation_embedding/embeddings*
_output_shapes

:*
dtype0
?
age_group_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name age_group_embedding/embeddings
?
2age_group_embedding/embeddings/Read/ReadVariableOpReadVariableOpage_group_embedding/embeddings*
_output_shapes

:*
dtype0
?
sex_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namesex_embedding/embeddings
?
,sex_embedding/embeddings/Read/ReadVariableOpReadVariableOpsex_embedding/embeddings*
_output_shapes

:*
dtype0
j

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name98*
value_dtype0	
l
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name55*
value_dtype0	
l
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name12*
value_dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:>*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:>>*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:>*
dtype0
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:>**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:>*
dtype0
?
(process_movie_embedding_with_genres/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*9
shared_name*(process_movie_embedding_with_genres/bias
?
<process_movie_embedding_with_genres/bias/Read/ReadVariableOpReadVariableOp(process_movie_embedding_with_genres/bias*
_output_shapes
:>*
dtype0
?
*process_movie_embedding_with_genres/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P>*;
shared_name,*process_movie_embedding_with_genres/kernel
?
>process_movie_embedding_with_genres/kernel/Read/ReadVariableOpReadVariableOp*process_movie_embedding_with_genres/kernel*
_output_shapes

:P>*
dtype0
?
genres_vector/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_namegenres_vector/embeddings
?
,genres_vector/embeddings/Read/ReadVariableOpReadVariableOpgenres_vector/embeddings*
_output_shapes
:	?*
dtype0
?
movie_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?>*+
shared_namemovie_embedding/embeddings
?
.movie_embedding/embeddings/Read/ReadVariableOpReadVariableOpmovie_embedding/embeddings*
_output_shapes
:	?>*
dtype0
m
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name147*
value_dtype0	
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?8
Const_1Const*
_output_shapes

:>*
dtype0*?8
value?8B?8>"?8?rF=??.<?????Լ??;m?D?7?F=?^C=? =?!=??<?ٱ<`_[;Xz???=]??_ɼpT*?v?!?hn+<??̼??<҃?<'?????=0?{?1z2=ijH=?M=mF????*=fp?<tgv<??&=??z???8?h?.??%׼?a=h?黺??<@?'<??????=@? ???=?0???D?;????C=?c?`A?Gt1?b= ・?V?0 ?;*?;??=??Ļ'lϼ ?????cm6=?L=???;?Z?<hB޻P??;}:=??S<?%???*??&=?
?:G?L?w< d:?I=Im=?^:?y{?@SO?&?	?D<0??&??<?|
=Вt;]?4?,???!w*=?1?<Gɨ?0x??`? c?-??X????:<:??<:W?<?j??9=N
?<?f&=!?=?'*??t0<??=?M?9=?=??H< ??:-?)=m?*???+=  ?;L?:??<??;?? =s???#$B?s?*=?U??s;su?? x^?z^?<???<?Ȼۭ<Sg/=̉s<?5???;;m?߼??	=??ܼLI5<?9K????@?+?c????	=i}=?(?;??=???<???<&??<??.???M??9???+?f4???= }?? ?C???=?(D?2?<?M|?M?C????<·?<gF?Z'?<??????ȼ???<??<n?<???c("=?í<g???-?ɼ?i?ZF??=?}j0??&??Y,???=?
$?3BH=?-#=?G?;?????f??
???=Ј>; s?? ???Q?;?n;;??3???B=??6=??ۼ??m???Zd??Ӭ=T??????pf$???L????;???w=???҆?<&a?<??_<0? ????;r??<?r"=@a????j????;???<?-??5#=???jE7?Р}?B3<?+?<  ?7??=G/=??7??|6=@?2?@?s??儼??"=????SYʼ??o<??= d?;???<?V5=(n<?8;?S??"9???6?+C=?zܼ-2??2?ӹ!?ל=`8Ѽ?-?;?b??F{?<??A?:??<L?v?`>;h}?; ?%???2; ?<B??<??A????48Y?@??ӏ=??G=@?ܼ=?9??4=??=Yk4=?Z	=?+?<??3=j?<@?+<???; ????<?????0??[???0;???<1?<L`#????;?.I=?x|<??;L@<?y?<"??< ??+ݻ?T;=?P= 4q????<[F?????p-;Xfu?!?F=?}#=34F=???????kC?J?&??2;????t@?Z??<s?,??0=???<[I?0%K;-???=?$?-?G?`]?t???1?< ?9@j??n?<?`F??.@????;?`?;?m??`?&?@2???M.=??	??#{<??;r?????????R??<c?(=lB??[-?yd=R͞<41&????S_????#=}v%=0?%;c?=??=2?<??t???@???=zc7?`"?????!= F??? =?V?:??:???<&?<P?-?z/???6??.,?@??:s?=?鄼?6???Y<?4????B??<v> ??0??'=72.?SD??c?=??;??Cv;Ɛ?<?7??P???+=ɳ=r9?<???`ӡ?0\d??c?<??5?P9???=&O?<?bἘ??;L?N?<?G?<???< f#???7???
<\?N<p?;??v>??ż?*=?????????/??mɼN=?<@􏼒??<??2;???<?9???"??v? ?J??m4??@lj??ۻ\?T<BR?<?x=PP?;F??<?Z?<??k;L?<???;?˧?}&
?)2='=?;=?Qֻ ???4?l<?????<???V4#???Լzy?<4t????؆???<?E?<غ6?L?<??;???<*2? i?G?C????=Z?B?牼??D?<p:???G??p?<???????rЃ<(???B???;Pg?;?*<?롼RӜ<?A??&0?w?3????<2L?<r??<@?5????<???;??!??	 =?<t?W<?p??#?<'?+=?,H=??E;??<ӥ=?4Qm<?"?<`?z? ??;???}*????@x???Q	????<A?=e????F;-S???<?5????+< ????R?;?e=#?=??=?h????:??<???r$?t&c??O#???=??J='???؋Z?t??<fx?<??<4?W<?1?Ǥ:=???<?w<?܁?I??q<?o????<?m2?PJ?;Ö???.??=S?=?i?<????`??:?a?Y?K=?Ħ?w-'=?f?? E;F?????=L????r;??H=?2?????@	ź??T? ?<?W?<Æ?j3???ƻ?_r<??-?Ǌܼ Կ:?J=?ۻ?Ĉ<N??<?P%???????<??;,???<?˻??&???˼0=d??=??c?=?ۻ?9???'=???}?#=?{]??=?Y??2?????<?d??y?==?65??1ƻ'$=`???????9?,=??ݼ\TY<-HE=???<?9R???>????;4-v<?G*??eɼ-Q??²?< ?<c&@? Ǽ.??<z??<m????<??<T3?? ??ƈ?<??=?3???%????<?pS?z?E???ļm6C??:???<?Z)<?)??z??^4="?<=f⎼?4м?滃?"??C?<?=?k?@????=???< $y:????0?(?0?^?amG=??<dC^<M3 =@?<t???L$'<????$
?]?==D?<Mt?? ?9??<???<?
=SX==y?=?=?.S?P*??!<i@?s???BK?<D?r<V???[r;???<0??????;L?"?(?n???J?"??<??:'?(=?I?Z ?<?:@<??ڻ?aҼ?v4??=??:?5??薼?=5=0??S?=?I=B>?< 0= ~?N??<??!<?Hļ?d=??*?-?%=?f޼̴s??3????.;?Y???,= ???F?<? K???:?Z??<?*(????<??=?l???c.??t*=Nf?<?G??u?< x???{H=sɻ???=???????????y?<0???&5?@q????D=4u6<M????Մ<B?< ??????<??ӼR
????<`???3<??<0Ғ;g?????,?S/???V<??g?)=??μ?hv????;???<2???;=`?W?'G???} ???.?!t=??0?(?M?B?D?M<??????<?fp???0=???<`O|;iB8=???<0"?;???<???;?????/=??a??3K??????(=}?&=$?y<??<?]H:?3*=?%=???;???Х
???????;@?U:?H?V
=`??:W?4?΍? iG??>??8?<??I???;??E=?<bb?<jCK??W@?"??<y?=?p?`g???U/= ????L;J?<},	????<0?2;g#??X <????V*	?&?<s)!=ZB????2A?<m?=??*?Q?/=??(=PD?;????<?S:=PCȻ??F?Bޔ<????`????ռ?I=?W?<a?; (
?a?=?P?C??`?I?]
'=F??<???`ʐ???F=] <Т??`*k??8?i./=?UA=Q??=J??G?<???ɻ????t?@< ???z?<?Q)=?=v??'@????;?Te?X9r????N??<?V??P?N?<??A??P?<@-???\.?`O???0?;,????m&=Z??<\???1<g?=?,8?gܼ?$=?0???^=g??????<???m'??MF=???<0|G?a@=?N?<@?躭?漖_)?r??<?z???????]?<??B<3?ۼ?A???e;4Ok<?????&=+=?=?&?&??<?Y=? <??޻=?(=?nF???<?G?<?N=??l?? ??.??<ڰ??η?<?D??:?ȼ?$A=?)@<z???-~A?S᩼1?4= dZ??L= ?o??K???4???V<?=:H9?(?<???<p?(?n??<gH?????\=??????%???.?F??<m????є???#=zӮ< ????ּ#P<?L=?Q2<??<?<:-J??c?;?`$;?-=???l?<?? =?'o<?I????<???<t9+<2Y?<??
=?KC=i?#=???:???#??z?˼ر*??mP;?6=??? =?? ?2?Р]?Z?6????<???:??H?v?&???K??\?<?tҼ???<A!8=t}8?.6=??!=?)??F?????<?(=?z??>'=n?<`?;wH??;?<?a?;-!=?86<?<??㼢Ø<]?C?F??<??=-18=??!??:o??!= ?);? ?:??S?ؼ??q;pF0?r??<??ϼ??<(<t???9W<???(& <?S??X??d???<?+?;??y?̴{<?>l?shͼ??Ӽh?߻????w?00?;?غ?5!=? =?SJ=Rˆ<??&?.*=??<v???P%=?A=?X?<?\??IR"=]m"=?= ???$?T<??E???0?h;?;}?$???ػ?g?<h?~?m՜?K?<sy/=")?<j? ?JڼG?? ?!?`?? @?; ûB?<??E??ǫ?4?????????;=о+??S??ã=ـ0=f`??St??.w?<(?'<6?` ??좼C!I=t?m??L?<󿽼gB? $?:R+?<:K?<??(?}= Lp?L????M???????r?3?=$??8K???F=???cK??Q??0?<?~???8??'=?P?vP$?4c3<?C???=?|'=??????f??%<???	?< ?G<0ѿ;ണ??|?@??->?J? ???5???K=?N???<???A=.j?<?<`?P???.?????h?????ռÒ??[?<f??<?Ȳ;P??;4Rr??:5=9B!=n??<??6=?n<??<.:?<?fc<Z=?<???:̘<?????Q<<(S3<??????.?2K?<????0?	??0?L?L????9?$??><?r4=??!=M/ؼ ?+<4g8<?H??4j?? ػ0?Ż?>@=-F.?2o?<???<(? ?&;R????-<??=?`??????	???Ǽ??;??????<?1?<?b==,???ŗ<??? h?W?=??E= ?<3=S?ļ?I?N??<?¶<?ܢ??!I?D
?gh-=-s=?T;=1J=!??4???&??<??<???<????\}s<???????sb??SL=????????L?<?70?2??<??<#?;=?j<? =????#????<???'<Z?????@< =Y!=-M ???;?D2=@??=?"??Q??G?м?6 ??K??2?;??~???˻??
??8	=?????*?X??f~?<m?:?3?-?'?,???=?v??s?ڼ?><?jƼ?Π?݇ =ј=B??<??< ???´?<??3'????<?N5<@@꼌2<??@= 68?P??;?G???No?y'?ͩ=Mּں?????<?9?<???@-?R??H?XTn??Xw???
??q?<4???????b???=?? =(*<3?/????<svE=}??6? ? ?̻Ӌ(?f(?<???????`?D???!??F=?
 =???<??<???ڻ?<mj?? ѓ??%??:?<?si<???9t?<t[&<'?:?O<?'?<0?0^;????YB=???<?Y????8<4?Z<???h?*?ݨ<=??,=??J????;M?K?h?_?mL߼Co?????n?<???m?????4??&?<?F#??8?< (S;s??:( ????<Ӄ??P??;??????<0 ??Kc<`?	??ˇ?????@???w?%=?8?<??G?4)<?8???????+<?N?z?F?:$????D=F?<6)=t!?m?:=?m?0?Q???7??<?'/;-?=͝?? ?4<?3?<Ͳڼ??传??: ?y?`'5;-??Ѓ???2?y?;=z`<X?3<d????B????<???<??=?1????U?<???<???<?H=?ޚ?i?=????P?ջ?=S<?=R?<????M?=?M??7?=?.1<@@??mI6? ];;??d<:C?Pj?;LO??ռ?'=ػ#?<="??<L?<?%?;?kC=?????-???=h?!<?qK=??=??;??7;g7D=0Բ??????<b??<?S?<??B?iX =*???*?S?Ӽtw????<?:<??<m*???"=0I?;s= ??;*?J?Ц?? h<z????9<????D?<?C <???w?=??2=???<?`!=Ţ??j)= }????ü????/!=??=?Tb??g??<??7=cRE?j?
? ??:?'=??8?#n"? ?*<???<???<N??<S62=??U???
???$?z??<??@=??;?3? ??*;??????;??=?U??ׄE=-???WI=?6?<?C+?&
?X>?b??<B?<???<??G= @{8?g???qQ??}!???=?{u???=?$<0vt? ??]vJ?
'G??IԼ)?7=????*$??>V?=?;Ѽr?;}B=?mH??e??Љ??Y???6? ?H<a4?'B??<=z????S??P?????M?͋μG?ȼД?;?V?<??ЕQ;?h?<?5= ??;?J<3?=?8=]??a?<?>?<??E;?!<0K??P&??{5=?K==?<?t???=?Iؼ:t??F??S캼1$=?"3???ۺL???SC=P0? ??R???f0=??A=??ļ\[Q<??????\lj<???<%b<'?A=?I?????a?<@?6???:???#?'??}B?#?8=̓9<?<??:岼?Ҳ<h???m?H=:?漦??<???<Q?<hN?;ݲ???B="?͒<??<L?`?ɺh?A<??<̹??b??<????=??E???F;0?9?:??<$??ZI?dEU<z&??D?p<12+=??\< ??;?&? ta8?{t?Qh)=?|?<@?<?b??'s???s=?D??D.=?z<? ҭ???=?B??`˛?ѝE=`?
??U?zX???!??FA??A=?b?<??????9=????{=???<4?N?Y?#=0{?;????" ?<	?=?kF=Y?????:????!????ټ ?s9?UI??uּߍ<???<??=?6 <4?0??(???C=
R
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
R
Const_4Const*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
??
Const_5Const*
_output_shapes	
:?*
dtype0*??
value??B???Bmovie_1Bmovie_2Bmovie_3Bmovie_4Bmovie_5Bmovie_6Bmovie_7Bmovie_8Bmovie_9Bmovie_10Bmovie_11Bmovie_12Bmovie_13Bmovie_14Bmovie_15Bmovie_16Bmovie_17Bmovie_18Bmovie_19Bmovie_20Bmovie_21Bmovie_22Bmovie_23Bmovie_24Bmovie_25Bmovie_26Bmovie_27Bmovie_28Bmovie_29Bmovie_30Bmovie_31Bmovie_32Bmovie_33Bmovie_34Bmovie_35Bmovie_36Bmovie_37Bmovie_38Bmovie_39Bmovie_40Bmovie_41Bmovie_42Bmovie_43Bmovie_44Bmovie_45Bmovie_46Bmovie_47Bmovie_48Bmovie_49Bmovie_50Bmovie_51Bmovie_52Bmovie_53Bmovie_54Bmovie_55Bmovie_56Bmovie_57Bmovie_58Bmovie_59Bmovie_60Bmovie_61Bmovie_62Bmovie_63Bmovie_64Bmovie_65Bmovie_66Bmovie_67Bmovie_68Bmovie_69Bmovie_70Bmovie_71Bmovie_72Bmovie_73Bmovie_74Bmovie_75Bmovie_76Bmovie_77Bmovie_78Bmovie_79Bmovie_80Bmovie_81Bmovie_82Bmovie_83Bmovie_84Bmovie_85Bmovie_86Bmovie_87Bmovie_88Bmovie_89Bmovie_90Bmovie_92Bmovie_93Bmovie_94Bmovie_95Bmovie_96Bmovie_97Bmovie_98Bmovie_99B	movie_100B	movie_101B	movie_102B	movie_103B	movie_104B	movie_105B	movie_106B	movie_107B	movie_108B	movie_109B	movie_110B	movie_111B	movie_112B	movie_113B	movie_114B	movie_115B	movie_116B	movie_117B	movie_118B	movie_119B	movie_120B	movie_121B	movie_122B	movie_123B	movie_124B	movie_125B	movie_126B	movie_127B	movie_128B	movie_129B	movie_130B	movie_131B	movie_132B	movie_133B	movie_134B	movie_135B	movie_136B	movie_137B	movie_138B	movie_139B	movie_140B	movie_141B	movie_142B	movie_143B	movie_144B	movie_145B	movie_146B	movie_147B	movie_148B	movie_149B	movie_150B	movie_151B	movie_152B	movie_153B	movie_154B	movie_155B	movie_156B	movie_157B	movie_158B	movie_159B	movie_160B	movie_161B	movie_162B	movie_163B	movie_164B	movie_165B	movie_166B	movie_167B	movie_168B	movie_169B	movie_170B	movie_171B	movie_172B	movie_173B	movie_174B	movie_175B	movie_176B	movie_177B	movie_178B	movie_179B	movie_180B	movie_181B	movie_182B	movie_183B	movie_184B	movie_185B	movie_186B	movie_187B	movie_188B	movie_189B	movie_190B	movie_191B	movie_192B	movie_193B	movie_194B	movie_195B	movie_196B	movie_197B	movie_198B	movie_199B	movie_200B	movie_201B	movie_202B	movie_203B	movie_204B	movie_205B	movie_206B	movie_207B	movie_208B	movie_209B	movie_210B	movie_211B	movie_212B	movie_213B	movie_214B	movie_215B	movie_216B	movie_217B	movie_218B	movie_219B	movie_220B	movie_222B	movie_223B	movie_224B	movie_225B	movie_226B	movie_227B	movie_228B	movie_229B	movie_230B	movie_231B	movie_232B	movie_233B	movie_234B	movie_235B	movie_236B	movie_237B	movie_238B	movie_239B	movie_240B	movie_241B	movie_242B	movie_243B	movie_244B	movie_245B	movie_246B	movie_247B	movie_248B	movie_249B	movie_250B	movie_251B	movie_252B	movie_253B	movie_254B	movie_255B	movie_256B	movie_257B	movie_258B	movie_259B	movie_260B	movie_261B	movie_262B	movie_263B	movie_264B	movie_265B	movie_266B	movie_267B	movie_268B	movie_269B	movie_270B	movie_271B	movie_272B	movie_273B	movie_274B	movie_275B	movie_276B	movie_277B	movie_278B	movie_279B	movie_280B	movie_281B	movie_282B	movie_283B	movie_284B	movie_285B	movie_286B	movie_287B	movie_288B	movie_289B	movie_290B	movie_291B	movie_292B	movie_293B	movie_294B	movie_295B	movie_296B	movie_297B	movie_298B	movie_299B	movie_300B	movie_301B	movie_302B	movie_303B	movie_304B	movie_305B	movie_306B	movie_307B	movie_308B	movie_309B	movie_310B	movie_311B	movie_312B	movie_313B	movie_314B	movie_315B	movie_316B	movie_317B	movie_318B	movie_319B	movie_320B	movie_321B	movie_322B	movie_324B	movie_325B	movie_326B	movie_327B	movie_328B	movie_329B	movie_330B	movie_331B	movie_332B	movie_333B	movie_334B	movie_335B	movie_336B	movie_337B	movie_338B	movie_339B	movie_340B	movie_341B	movie_342B	movie_343B	movie_344B	movie_345B	movie_346B	movie_347B	movie_348B	movie_349B	movie_350B	movie_351B	movie_352B	movie_353B	movie_354B	movie_355B	movie_356B	movie_357B	movie_358B	movie_359B	movie_360B	movie_361B	movie_362B	movie_363B	movie_364B	movie_365B	movie_366B	movie_367B	movie_368B	movie_369B	movie_370B	movie_371B	movie_372B	movie_373B	movie_374B	movie_375B	movie_376B	movie_377B	movie_378B	movie_379B	movie_380B	movie_381B	movie_382B	movie_383B	movie_384B	movie_385B	movie_386B	movie_387B	movie_388B	movie_389B	movie_390B	movie_391B	movie_392B	movie_393B	movie_394B	movie_395B	movie_396B	movie_397B	movie_398B	movie_399B	movie_400B	movie_401B	movie_402B	movie_403B	movie_404B	movie_405B	movie_406B	movie_407B	movie_408B	movie_409B	movie_410B	movie_411B	movie_412B	movie_413B	movie_414B	movie_415B	movie_416B	movie_417B	movie_418B	movie_419B	movie_420B	movie_421B	movie_422B	movie_423B	movie_424B	movie_425B	movie_426B	movie_427B	movie_428B	movie_429B	movie_430B	movie_431B	movie_432B	movie_433B	movie_434B	movie_435B	movie_436B	movie_437B	movie_438B	movie_439B	movie_440B	movie_441B	movie_442B	movie_443B	movie_444B	movie_445B	movie_446B	movie_447B	movie_448B	movie_449B	movie_450B	movie_451B	movie_452B	movie_453B	movie_454B	movie_455B	movie_456B	movie_457B	movie_458B	movie_459B	movie_460B	movie_461B	movie_462B	movie_463B	movie_464B	movie_465B	movie_466B	movie_467B	movie_468B	movie_469B	movie_470B	movie_471B	movie_472B	movie_473B	movie_474B	movie_475B	movie_476B	movie_477B	movie_478B	movie_479B	movie_480B	movie_481B	movie_482B	movie_483B	movie_484B	movie_485B	movie_486B	movie_487B	movie_488B	movie_489B	movie_490B	movie_491B	movie_492B	movie_493B	movie_494B	movie_495B	movie_496B	movie_497B	movie_498B	movie_499B	movie_500B	movie_501B	movie_502B	movie_503B	movie_504B	movie_505B	movie_506B	movie_507B	movie_508B	movie_509B	movie_510B	movie_511B	movie_512B	movie_513B	movie_514B	movie_515B	movie_516B	movie_517B	movie_518B	movie_519B	movie_520B	movie_521B	movie_522B	movie_523B	movie_524B	movie_525B	movie_526B	movie_527B	movie_528B	movie_529B	movie_530B	movie_531B	movie_532B	movie_533B	movie_534B	movie_535B	movie_536B	movie_537B	movie_538B	movie_539B	movie_540B	movie_541B	movie_542B	movie_543B	movie_544B	movie_545B	movie_546B	movie_547B	movie_548B	movie_549B	movie_550B	movie_551B	movie_552B	movie_553B	movie_554B	movie_555B	movie_556B	movie_557B	movie_558B	movie_559B	movie_560B	movie_561B	movie_562B	movie_563B	movie_564B	movie_565B	movie_566B	movie_567B	movie_568B	movie_569B	movie_570B	movie_571B	movie_572B	movie_573B	movie_574B	movie_575B	movie_576B	movie_577B	movie_578B	movie_579B	movie_580B	movie_581B	movie_582B	movie_583B	movie_584B	movie_585B	movie_586B	movie_587B	movie_588B	movie_589B	movie_590B	movie_591B	movie_592B	movie_593B	movie_594B	movie_595B	movie_596B	movie_597B	movie_598B	movie_599B	movie_600B	movie_601B	movie_602B	movie_603B	movie_604B	movie_605B	movie_606B	movie_607B	movie_608B	movie_609B	movie_610B	movie_611B	movie_612B	movie_613B	movie_614B	movie_615B	movie_616B	movie_617B	movie_618B	movie_619B	movie_620B	movie_621B	movie_623B	movie_624B	movie_625B	movie_626B	movie_627B	movie_628B	movie_629B	movie_630B	movie_631B	movie_632B	movie_633B	movie_634B	movie_635B	movie_636B	movie_637B	movie_638B	movie_639B	movie_640B	movie_641B	movie_642B	movie_643B	movie_644B	movie_645B	movie_647B	movie_648B	movie_649B	movie_650B	movie_651B	movie_652B	movie_653B	movie_654B	movie_655B	movie_656B	movie_657B	movie_658B	movie_659B	movie_660B	movie_661B	movie_662B	movie_663B	movie_664B	movie_665B	movie_666B	movie_667B	movie_668B	movie_669B	movie_670B	movie_671B	movie_672B	movie_673B	movie_674B	movie_675B	movie_676B	movie_678B	movie_679B	movie_680B	movie_681B	movie_682B	movie_683B	movie_684B	movie_685B	movie_687B	movie_688B	movie_690B	movie_691B	movie_692B	movie_693B	movie_694B	movie_695B	movie_696B	movie_697B	movie_698B	movie_699B	movie_700B	movie_701B	movie_702B	movie_703B	movie_704B	movie_705B	movie_706B	movie_707B	movie_708B	movie_709B	movie_710B	movie_711B	movie_712B	movie_713B	movie_714B	movie_715B	movie_716B	movie_717B	movie_718B	movie_719B	movie_720B	movie_721B	movie_722B	movie_723B	movie_724B	movie_725B	movie_726B	movie_727B	movie_728B	movie_729B	movie_730B	movie_731B	movie_732B	movie_733B	movie_734B	movie_735B	movie_736B	movie_737B	movie_738B	movie_739B	movie_741B	movie_742B	movie_743B	movie_744B	movie_745B	movie_746B	movie_747B	movie_748B	movie_749B	movie_750B	movie_751B	movie_752B	movie_753B	movie_754B	movie_755B	movie_756B	movie_757B	movie_758B	movie_759B	movie_760B	movie_761B	movie_762B	movie_763B	movie_764B	movie_765B	movie_766B	movie_767B	movie_768B	movie_769B	movie_770B	movie_771B	movie_772B	movie_773B	movie_774B	movie_775B	movie_776B	movie_777B	movie_778B	movie_779B	movie_780B	movie_781B	movie_782B	movie_783B	movie_784B	movie_785B	movie_786B	movie_787B	movie_788B	movie_789B	movie_790B	movie_791B	movie_792B	movie_793B	movie_794B	movie_795B	movie_796B	movie_797B	movie_798B	movie_799B	movie_800B	movie_801B	movie_802B	movie_803B	movie_804B	movie_805B	movie_806B	movie_807B	movie_808B	movie_809B	movie_810B	movie_811B	movie_812B	movie_813B	movie_814B	movie_815B	movie_816B	movie_818B	movie_819B	movie_820B	movie_821B	movie_822B	movie_823B	movie_824B	movie_825B	movie_826B	movie_827B	movie_828B	movie_829B	movie_830B	movie_831B	movie_832B	movie_833B	movie_834B	movie_835B	movie_836B	movie_837B	movie_838B	movie_839B	movie_840B	movie_841B	movie_842B	movie_843B	movie_844B	movie_845B	movie_846B	movie_847B	movie_848B	movie_849B	movie_850B	movie_851B	movie_852B	movie_853B	movie_854B	movie_855B	movie_856B	movie_857B	movie_858B	movie_859B	movie_860B	movie_861B	movie_862B	movie_863B	movie_864B	movie_865B	movie_866B	movie_867B	movie_868B	movie_869B	movie_870B	movie_871B	movie_872B	movie_873B	movie_874B	movie_875B	movie_876B	movie_877B	movie_878B	movie_879B	movie_880B	movie_881B	movie_882B	movie_884B	movie_885B	movie_886B	movie_887B	movie_888B	movie_889B	movie_890B	movie_891B	movie_892B	movie_893B	movie_894B	movie_895B	movie_896B	movie_897B	movie_898B	movie_899B	movie_900B	movie_901B	movie_902B	movie_903B	movie_904B	movie_905B	movie_906B	movie_907B	movie_908B	movie_909B	movie_910B	movie_911B	movie_912B	movie_913B	movie_914B	movie_915B	movie_916B	movie_917B	movie_918B	movie_919B	movie_920B	movie_921B	movie_922B	movie_923B	movie_924B	movie_925B	movie_926B	movie_927B	movie_928B	movie_929B	movie_930B	movie_931B	movie_932B	movie_933B	movie_934B	movie_935B	movie_936B	movie_937B	movie_938B	movie_939B	movie_940B	movie_941B	movie_942B	movie_943B	movie_944B	movie_945B	movie_946B	movie_947B	movie_948B	movie_949B	movie_950B	movie_951B	movie_952B	movie_953B	movie_954B	movie_955B	movie_956B	movie_957B	movie_958B	movie_959B	movie_960B	movie_961B	movie_962B	movie_963B	movie_964B	movie_965B	movie_966B	movie_967B	movie_968B	movie_969B	movie_970B	movie_971B	movie_972B	movie_973B	movie_974B	movie_975B	movie_976B	movie_977B	movie_978B	movie_979B	movie_980B	movie_981B	movie_982B	movie_983B	movie_984B	movie_985B	movie_986B	movie_987B	movie_988B	movie_989B	movie_990B	movie_991B	movie_992B	movie_993B	movie_994B	movie_996B	movie_997B	movie_998B	movie_999B
movie_1000B
movie_1001B
movie_1002B
movie_1003B
movie_1004B
movie_1005B
movie_1006B
movie_1007B
movie_1008B
movie_1009B
movie_1010B
movie_1011B
movie_1012B
movie_1013B
movie_1014B
movie_1015B
movie_1016B
movie_1017B
movie_1018B
movie_1019B
movie_1020B
movie_1021B
movie_1022B
movie_1023B
movie_1024B
movie_1025B
movie_1026B
movie_1027B
movie_1028B
movie_1029B
movie_1030B
movie_1031B
movie_1032B
movie_1033B
movie_1034B
movie_1035B
movie_1036B
movie_1037B
movie_1038B
movie_1039B
movie_1040B
movie_1041B
movie_1042B
movie_1043B
movie_1044B
movie_1045B
movie_1046B
movie_1047B
movie_1049B
movie_1050B
movie_1051B
movie_1052B
movie_1053B
movie_1054B
movie_1055B
movie_1056B
movie_1057B
movie_1058B
movie_1059B
movie_1060B
movie_1061B
movie_1062B
movie_1063B
movie_1064B
movie_1065B
movie_1066B
movie_1067B
movie_1068B
movie_1069B
movie_1070B
movie_1071B
movie_1073B
movie_1075B
movie_1076B
movie_1077B
movie_1078B
movie_1079B
movie_1080B
movie_1081B
movie_1082B
movie_1083B
movie_1084B
movie_1085B
movie_1086B
movie_1087B
movie_1088B
movie_1089B
movie_1090B
movie_1091B
movie_1092B
movie_1093B
movie_1094B
movie_1095B
movie_1096B
movie_1097B
movie_1098B
movie_1099B
movie_1100B
movie_1101B
movie_1102B
movie_1103B
movie_1104B
movie_1105B
movie_1106B
movie_1107B
movie_1108B
movie_1109B
movie_1110B
movie_1111B
movie_1112B
movie_1113B
movie_1114B
movie_1115B
movie_1116B
movie_1117B
movie_1118B
movie_1119B
movie_1120B
movie_1121B
movie_1122B
movie_1123B
movie_1124B
movie_1125B
movie_1126B
movie_1127B
movie_1128B
movie_1129B
movie_1130B
movie_1131B
movie_1132B
movie_1133B
movie_1134B
movie_1135B
movie_1136B
movie_1137B
movie_1138B
movie_1139B
movie_1140B
movie_1141B
movie_1142B
movie_1143B
movie_1144B
movie_1145B
movie_1146B
movie_1147B
movie_1148B
movie_1149B
movie_1150B
movie_1151B
movie_1152B
movie_1153B
movie_1154B
movie_1155B
movie_1156B
movie_1157B
movie_1158B
movie_1159B
movie_1160B
movie_1161B
movie_1162B
movie_1163B
movie_1164B
movie_1165B
movie_1166B
movie_1167B
movie_1168B
movie_1169B
movie_1170B
movie_1171B
movie_1172B
movie_1173B
movie_1174B
movie_1175B
movie_1176B
movie_1177B
movie_1178B
movie_1179B
movie_1180B
movie_1181B
movie_1183B
movie_1184B
movie_1185B
movie_1186B
movie_1187B
movie_1188B
movie_1189B
movie_1190B
movie_1191B
movie_1192B
movie_1193B
movie_1194B
movie_1196B
movie_1197B
movie_1198B
movie_1199B
movie_1200B
movie_1201B
movie_1202B
movie_1203B
movie_1204B
movie_1205B
movie_1206B
movie_1207B
movie_1208B
movie_1209B
movie_1210B
movie_1211B
movie_1212B
movie_1213B
movie_1214B
movie_1215B
movie_1216B
movie_1217B
movie_1218B
movie_1219B
movie_1220B
movie_1221B
movie_1222B
movie_1223B
movie_1224B
movie_1225B
movie_1226B
movie_1227B
movie_1228B
movie_1230B
movie_1231B
movie_1232B
movie_1233B
movie_1234B
movie_1235B
movie_1236B
movie_1237B
movie_1238B
movie_1240B
movie_1241B
movie_1242B
movie_1243B
movie_1244B
movie_1245B
movie_1246B
movie_1247B
movie_1248B
movie_1249B
movie_1250B
movie_1251B
movie_1252B
movie_1253B
movie_1254B
movie_1255B
movie_1256B
movie_1257B
movie_1258B
movie_1259B
movie_1260B
movie_1261B
movie_1262B
movie_1263B
movie_1264B
movie_1265B
movie_1266B
movie_1267B
movie_1268B
movie_1269B
movie_1270B
movie_1271B
movie_1272B
movie_1273B
movie_1274B
movie_1275B
movie_1276B
movie_1277B
movie_1278B
movie_1279B
movie_1280B
movie_1281B
movie_1282B
movie_1283B
movie_1284B
movie_1285B
movie_1286B
movie_1287B
movie_1288B
movie_1289B
movie_1290B
movie_1291B
movie_1292B
movie_1293B
movie_1294B
movie_1295B
movie_1296B
movie_1297B
movie_1298B
movie_1299B
movie_1300B
movie_1301B
movie_1302B
movie_1303B
movie_1304B
movie_1305B
movie_1306B
movie_1307B
movie_1308B
movie_1309B
movie_1310B
movie_1311B
movie_1312B
movie_1313B
movie_1314B
movie_1315B
movie_1316B
movie_1317B
movie_1318B
movie_1319B
movie_1320B
movie_1321B
movie_1322B
movie_1323B
movie_1324B
movie_1325B
movie_1326B
movie_1327B
movie_1328B
movie_1329B
movie_1330B
movie_1331B
movie_1332B
movie_1333B
movie_1334B
movie_1335B
movie_1336B
movie_1337B
movie_1339B
movie_1340B
movie_1341B
movie_1342B
movie_1343B
movie_1344B
movie_1345B
movie_1346B
movie_1347B
movie_1348B
movie_1349B
movie_1350B
movie_1351B
movie_1352B
movie_1353B
movie_1354B
movie_1355B
movie_1356B
movie_1357B
movie_1358B
movie_1359B
movie_1360B
movie_1361B
movie_1362B
movie_1363B
movie_1364B
movie_1365B
movie_1366B
movie_1367B
movie_1368B
movie_1369B
movie_1370B
movie_1371B
movie_1372B
movie_1373B
movie_1374B
movie_1375B
movie_1376B
movie_1377B
movie_1378B
movie_1379B
movie_1380B
movie_1381B
movie_1382B
movie_1383B
movie_1384B
movie_1385B
movie_1386B
movie_1387B
movie_1388B
movie_1389B
movie_1390B
movie_1391B
movie_1392B
movie_1393B
movie_1394B
movie_1395B
movie_1396B
movie_1397B
movie_1398B
movie_1399B
movie_1400B
movie_1401B
movie_1404B
movie_1405B
movie_1406B
movie_1407B
movie_1408B
movie_1409B
movie_1410B
movie_1411B
movie_1412B
movie_1413B
movie_1414B
movie_1415B
movie_1416B
movie_1417B
movie_1419B
movie_1420B
movie_1421B
movie_1422B
movie_1423B
movie_1424B
movie_1425B
movie_1426B
movie_1427B
movie_1428B
movie_1429B
movie_1430B
movie_1431B
movie_1432B
movie_1433B
movie_1434B
movie_1436B
movie_1437B
movie_1438B
movie_1439B
movie_1440B
movie_1441B
movie_1442B
movie_1443B
movie_1444B
movie_1445B
movie_1446B
movie_1447B
movie_1448B
movie_1449B
movie_1450B
movie_1453B
movie_1454B
movie_1455B
movie_1456B
movie_1457B
movie_1458B
movie_1459B
movie_1460B
movie_1461B
movie_1462B
movie_1463B
movie_1464B
movie_1465B
movie_1466B
movie_1467B
movie_1468B
movie_1470B
movie_1471B
movie_1472B
movie_1473B
movie_1474B
movie_1475B
movie_1476B
movie_1477B
movie_1479B
movie_1480B
movie_1482B
movie_1483B
movie_1484B
movie_1485B
movie_1486B
movie_1487B
movie_1488B
movie_1489B
movie_1490B
movie_1493B
movie_1494B
movie_1495B
movie_1496B
movie_1497B
movie_1498B
movie_1499B
movie_1500B
movie_1501B
movie_1502B
movie_1503B
movie_1504B
movie_1507B
movie_1508B
movie_1509B
movie_1510B
movie_1511B
movie_1513B
movie_1514B
movie_1515B
movie_1516B
movie_1517B
movie_1518B
movie_1519B
movie_1520B
movie_1522B
movie_1523B
movie_1524B
movie_1525B
movie_1526B
movie_1527B
movie_1528B
movie_1529B
movie_1531B
movie_1532B
movie_1533B
movie_1534B
movie_1535B
movie_1537B
movie_1538B
movie_1539B
movie_1541B
movie_1542B
movie_1543B
movie_1544B
movie_1545B
movie_1546B
movie_1547B
movie_1548B
movie_1549B
movie_1550B
movie_1551B
movie_1552B
movie_1553B
movie_1554B
movie_1555B
movie_1556B
movie_1557B
movie_1558B
movie_1559B
movie_1561B
movie_1562B
movie_1563B
movie_1564B
movie_1565B
movie_1566B
movie_1567B
movie_1568B
movie_1569B
movie_1570B
movie_1571B
movie_1572B
movie_1573B
movie_1574B
movie_1575B
movie_1577B
movie_1578B
movie_1579B
movie_1580B
movie_1581B
movie_1582B
movie_1583B
movie_1584B
movie_1585B
movie_1586B
movie_1587B
movie_1588B
movie_1589B
movie_1590B
movie_1591B
movie_1592B
movie_1593B
movie_1594B
movie_1595B
movie_1596B
movie_1597B
movie_1598B
movie_1599B
movie_1600B
movie_1601B
movie_1602B
movie_1603B
movie_1604B
movie_1605B
movie_1606B
movie_1608B
movie_1609B
movie_1610B
movie_1611B
movie_1612B
movie_1613B
movie_1614B
movie_1615B
movie_1616B
movie_1617B
movie_1619B
movie_1620B
movie_1621B
movie_1622B
movie_1623B
movie_1624B
movie_1625B
movie_1626B
movie_1627B
movie_1628B
movie_1629B
movie_1630B
movie_1631B
movie_1632B
movie_1633B
movie_1635B
movie_1636B
movie_1639B
movie_1640B
movie_1641B
movie_1642B
movie_1643B
movie_1644B
movie_1645B
movie_1646B
movie_1647B
movie_1648B
movie_1649B
movie_1650B
movie_1651B
movie_1652B
movie_1653B
movie_1654B
movie_1655B
movie_1656B
movie_1657B
movie_1658B
movie_1659B
movie_1660B
movie_1661B
movie_1662B
movie_1663B
movie_1664B
movie_1665B
movie_1666B
movie_1667B
movie_1668B
movie_1669B
movie_1670B
movie_1671B
movie_1672B
movie_1673B
movie_1674B
movie_1675B
movie_1676B
movie_1677B
movie_1678B
movie_1679B
movie_1680B
movie_1681B
movie_1682B
movie_1683B
movie_1684B
movie_1685B
movie_1686B
movie_1687B
movie_1688B
movie_1689B
movie_1690B
movie_1692B
movie_1693B
movie_1694B
movie_1695B
movie_1696B
movie_1697B
movie_1698B
movie_1699B
movie_1701B
movie_1702B
movie_1703B
movie_1704B
movie_1705B
movie_1706B
movie_1707B
movie_1708B
movie_1709B
movie_1710B
movie_1711B
movie_1713B
movie_1714B
movie_1715B
movie_1716B
movie_1717B
movie_1718B
movie_1719B
movie_1720B
movie_1721B
movie_1722B
movie_1723B
movie_1724B
movie_1725B
movie_1726B
movie_1727B
movie_1728B
movie_1729B
movie_1730B
movie_1731B
movie_1732B
movie_1733B
movie_1734B
movie_1735B
movie_1738B
movie_1739B
movie_1740B
movie_1741B
movie_1742B
movie_1743B
movie_1744B
movie_1746B
movie_1747B
movie_1748B
movie_1749B
movie_1750B
movie_1752B
movie_1753B
movie_1754B
movie_1755B
movie_1756B
movie_1757B
movie_1758B
movie_1759B
movie_1760B
movie_1762B
movie_1764B
movie_1765B
movie_1767B
movie_1768B
movie_1769B
movie_1770B
movie_1771B
movie_1772B
movie_1773B
movie_1774B
movie_1776B
movie_1777B
movie_1779B
movie_1780B
movie_1781B
movie_1782B
movie_1783B
movie_1784B
movie_1785B
movie_1787B
movie_1788B
movie_1789B
movie_1791B
movie_1792B
movie_1793B
movie_1794B
movie_1795B
movie_1796B
movie_1797B
movie_1798B
movie_1799B
movie_1801B
movie_1804B
movie_1805B
movie_1806B
movie_1807B
movie_1809B
movie_1810B
movie_1811B
movie_1812B
movie_1814B
movie_1815B
movie_1816B
movie_1817B
movie_1819B
movie_1820B
movie_1821B
movie_1822B
movie_1824B
movie_1825B
movie_1826B
movie_1827B
movie_1829B
movie_1830B
movie_1831B
movie_1832B
movie_1833B
movie_1834B
movie_1835B
movie_1836B
movie_1837B
movie_1839B
movie_1840B
movie_1841B
movie_1842B
movie_1843B
movie_1844B
movie_1845B
movie_1846B
movie_1847B
movie_1848B
movie_1849B
movie_1850B
movie_1851B
movie_1852B
movie_1853B
movie_1854B
movie_1855B
movie_1856B
movie_1857B
movie_1858B
movie_1859B
movie_1860B
movie_1861B
movie_1862B
movie_1863B
movie_1864B
movie_1865B
movie_1866B
movie_1867B
movie_1868B
movie_1869B
movie_1870B
movie_1871B
movie_1872B
movie_1873B
movie_1874B
movie_1875B
movie_1876B
movie_1877B
movie_1878B
movie_1879B
movie_1880B
movie_1881B
movie_1882B
movie_1883B
movie_1884B
movie_1885B
movie_1886B
movie_1887B
movie_1888B
movie_1889B
movie_1890B
movie_1891B
movie_1892B
movie_1893B
movie_1894B
movie_1895B
movie_1896B
movie_1897B
movie_1898B
movie_1899B
movie_1900B
movie_1901B
movie_1902B
movie_1903B
movie_1904B
movie_1905B
movie_1906B
movie_1907B
movie_1908B
movie_1909B
movie_1910B
movie_1911B
movie_1912B
movie_1913B
movie_1914B
movie_1915B
movie_1916B
movie_1917B
movie_1918B
movie_1919B
movie_1920B
movie_1921B
movie_1922B
movie_1923B
movie_1924B
movie_1925B
movie_1926B
movie_1927B
movie_1928B
movie_1929B
movie_1930B
movie_1931B
movie_1932B
movie_1933B
movie_1934B
movie_1935B
movie_1936B
movie_1937B
movie_1938B
movie_1939B
movie_1940B
movie_1941B
movie_1942B
movie_1943B
movie_1944B
movie_1945B
movie_1946B
movie_1947B
movie_1948B
movie_1949B
movie_1950B
movie_1951B
movie_1952B
movie_1953B
movie_1954B
movie_1955B
movie_1956B
movie_1957B
movie_1958B
movie_1959B
movie_1960B
movie_1961B
movie_1962B
movie_1963B
movie_1964B
movie_1965B
movie_1966B
movie_1967B
movie_1968B
movie_1969B
movie_1970B
movie_1971B
movie_1972B
movie_1973B
movie_1974B
movie_1975B
movie_1976B
movie_1977B
movie_1978B
movie_1979B
movie_1980B
movie_1981B
movie_1982B
movie_1983B
movie_1984B
movie_1985B
movie_1986B
movie_1987B
movie_1988B
movie_1989B
movie_1990B
movie_1991B
movie_1992B
movie_1993B
movie_1994B
movie_1995B
movie_1996B
movie_1997B
movie_1998B
movie_1999B
movie_2000B
movie_2001B
movie_2002B
movie_2003B
movie_2004B
movie_2005B
movie_2006B
movie_2007B
movie_2008B
movie_2009B
movie_2010B
movie_2011B
movie_2012B
movie_2013B
movie_2014B
movie_2015B
movie_2016B
movie_2017B
movie_2018B
movie_2019B
movie_2020B
movie_2021B
movie_2022B
movie_2023B
movie_2024B
movie_2025B
movie_2026B
movie_2027B
movie_2028B
movie_2029B
movie_2030B
movie_2031B
movie_2032B
movie_2033B
movie_2034B
movie_2035B
movie_2036B
movie_2037B
movie_2038B
movie_2039B
movie_2040B
movie_2041B
movie_2042B
movie_2043B
movie_2044B
movie_2045B
movie_2046B
movie_2047B
movie_2048B
movie_2049B
movie_2050B
movie_2051B
movie_2052B
movie_2053B
movie_2054B
movie_2055B
movie_2056B
movie_2057B
movie_2058B
movie_2059B
movie_2060B
movie_2061B
movie_2062B
movie_2063B
movie_2064B
movie_2065B
movie_2066B
movie_2067B
movie_2068B
movie_2069B
movie_2070B
movie_2071B
movie_2072B
movie_2073B
movie_2074B
movie_2075B
movie_2076B
movie_2077B
movie_2078B
movie_2079B
movie_2080B
movie_2081B
movie_2082B
movie_2083B
movie_2084B
movie_2085B
movie_2086B
movie_2087B
movie_2088B
movie_2089B
movie_2090B
movie_2091B
movie_2092B
movie_2093B
movie_2094B
movie_2095B
movie_2096B
movie_2097B
movie_2098B
movie_2099B
movie_2100B
movie_2101B
movie_2102B
movie_2103B
movie_2104B
movie_2105B
movie_2106B
movie_2107B
movie_2108B
movie_2109B
movie_2110B
movie_2111B
movie_2112B
movie_2113B
movie_2114B
movie_2115B
movie_2116B
movie_2117B
movie_2118B
movie_2119B
movie_2120B
movie_2121B
movie_2122B
movie_2123B
movie_2124B
movie_2125B
movie_2126B
movie_2127B
movie_2128B
movie_2129B
movie_2130B
movie_2131B
movie_2132B
movie_2133B
movie_2134B
movie_2135B
movie_2136B
movie_2137B
movie_2138B
movie_2139B
movie_2140B
movie_2141B
movie_2142B
movie_2143B
movie_2144B
movie_2145B
movie_2146B
movie_2147B
movie_2148B
movie_2149B
movie_2150B
movie_2151B
movie_2152B
movie_2153B
movie_2154B
movie_2155B
movie_2156B
movie_2157B
movie_2158B
movie_2159B
movie_2160B
movie_2161B
movie_2162B
movie_2163B
movie_2164B
movie_2165B
movie_2166B
movie_2167B
movie_2168B
movie_2169B
movie_2170B
movie_2171B
movie_2172B
movie_2173B
movie_2174B
movie_2175B
movie_2176B
movie_2177B
movie_2178B
movie_2179B
movie_2180B
movie_2181B
movie_2182B
movie_2183B
movie_2184B
movie_2185B
movie_2186B
movie_2187B
movie_2188B
movie_2189B
movie_2190B
movie_2191B
movie_2192B
movie_2193B
movie_2194B
movie_2195B
movie_2196B
movie_2197B
movie_2198B
movie_2199B
movie_2200B
movie_2201B
movie_2202B
movie_2203B
movie_2204B
movie_2205B
movie_2206B
movie_2207B
movie_2208B
movie_2209B
movie_2210B
movie_2211B
movie_2212B
movie_2213B
movie_2214B
movie_2215B
movie_2216B
movie_2217B
movie_2218B
movie_2219B
movie_2220B
movie_2221B
movie_2222B
movie_2223B
movie_2224B
movie_2225B
movie_2226B
movie_2227B
movie_2228B
movie_2229B
movie_2230B
movie_2231B
movie_2232B
movie_2233B
movie_2234B
movie_2235B
movie_2236B
movie_2237B
movie_2238B
movie_2239B
movie_2240B
movie_2241B
movie_2242B
movie_2243B
movie_2244B
movie_2245B
movie_2246B
movie_2247B
movie_2248B
movie_2249B
movie_2250B
movie_2251B
movie_2252B
movie_2253B
movie_2254B
movie_2255B
movie_2256B
movie_2257B
movie_2258B
movie_2259B
movie_2260B
movie_2261B
movie_2262B
movie_2263B
movie_2264B
movie_2265B
movie_2266B
movie_2267B
movie_2268B
movie_2269B
movie_2270B
movie_2271B
movie_2272B
movie_2273B
movie_2274B
movie_2275B
movie_2276B
movie_2277B
movie_2278B
movie_2279B
movie_2280B
movie_2281B
movie_2282B
movie_2283B
movie_2284B
movie_2285B
movie_2286B
movie_2287B
movie_2288B
movie_2289B
movie_2290B
movie_2291B
movie_2292B
movie_2293B
movie_2294B
movie_2295B
movie_2296B
movie_2297B
movie_2298B
movie_2299B
movie_2300B
movie_2301B
movie_2302B
movie_2303B
movie_2304B
movie_2305B
movie_2306B
movie_2307B
movie_2308B
movie_2309B
movie_2310B
movie_2311B
movie_2312B
movie_2313B
movie_2314B
movie_2315B
movie_2316B
movie_2317B
movie_2318B
movie_2319B
movie_2320B
movie_2321B
movie_2322B
movie_2323B
movie_2324B
movie_2325B
movie_2326B
movie_2327B
movie_2328B
movie_2329B
movie_2330B
movie_2331B
movie_2332B
movie_2333B
movie_2334B
movie_2335B
movie_2336B
movie_2337B
movie_2338B
movie_2339B
movie_2340B
movie_2341B
movie_2342B
movie_2343B
movie_2344B
movie_2345B
movie_2346B
movie_2347B
movie_2348B
movie_2349B
movie_2350B
movie_2351B
movie_2352B
movie_2353B
movie_2354B
movie_2355B
movie_2356B
movie_2357B
movie_2358B
movie_2359B
movie_2360B
movie_2361B
movie_2362B
movie_2363B
movie_2364B
movie_2365B
movie_2366B
movie_2367B
movie_2368B
movie_2369B
movie_2370B
movie_2371B
movie_2372B
movie_2373B
movie_2374B
movie_2375B
movie_2376B
movie_2377B
movie_2378B
movie_2379B
movie_2380B
movie_2381B
movie_2382B
movie_2383B
movie_2384B
movie_2385B
movie_2386B
movie_2387B
movie_2388B
movie_2389B
movie_2390B
movie_2391B
movie_2392B
movie_2393B
movie_2394B
movie_2395B
movie_2396B
movie_2397B
movie_2398B
movie_2399B
movie_2400B
movie_2401B
movie_2402B
movie_2403B
movie_2404B
movie_2405B
movie_2406B
movie_2407B
movie_2408B
movie_2409B
movie_2410B
movie_2411B
movie_2412B
movie_2413B
movie_2414B
movie_2415B
movie_2416B
movie_2417B
movie_2418B
movie_2419B
movie_2420B
movie_2421B
movie_2422B
movie_2423B
movie_2424B
movie_2425B
movie_2426B
movie_2427B
movie_2428B
movie_2429B
movie_2430B
movie_2431B
movie_2432B
movie_2433B
movie_2434B
movie_2435B
movie_2436B
movie_2437B
movie_2438B
movie_2439B
movie_2440B
movie_2441B
movie_2442B
movie_2443B
movie_2444B
movie_2445B
movie_2446B
movie_2447B
movie_2448B
movie_2449B
movie_2450B
movie_2451B
movie_2452B
movie_2453B
movie_2454B
movie_2455B
movie_2456B
movie_2457B
movie_2458B
movie_2459B
movie_2460B
movie_2461B
movie_2462B
movie_2463B
movie_2464B
movie_2465B
movie_2466B
movie_2467B
movie_2468B
movie_2469B
movie_2470B
movie_2471B
movie_2472B
movie_2473B
movie_2474B
movie_2475B
movie_2476B
movie_2477B
movie_2478B
movie_2479B
movie_2480B
movie_2481B
movie_2482B
movie_2483B
movie_2484B
movie_2485B
movie_2486B
movie_2487B
movie_2488B
movie_2489B
movie_2490B
movie_2491B
movie_2492B
movie_2493B
movie_2494B
movie_2495B
movie_2496B
movie_2497B
movie_2498B
movie_2499B
movie_2500B
movie_2501B
movie_2502B
movie_2503B
movie_2504B
movie_2505B
movie_2506B
movie_2507B
movie_2508B
movie_2509B
movie_2510B
movie_2511B
movie_2512B
movie_2513B
movie_2514B
movie_2515B
movie_2516B
movie_2517B
movie_2518B
movie_2519B
movie_2520B
movie_2521B
movie_2522B
movie_2523B
movie_2524B
movie_2525B
movie_2526B
movie_2527B
movie_2528B
movie_2529B
movie_2530B
movie_2531B
movie_2532B
movie_2533B
movie_2534B
movie_2535B
movie_2536B
movie_2537B
movie_2538B
movie_2539B
movie_2540B
movie_2541B
movie_2542B
movie_2543B
movie_2544B
movie_2545B
movie_2546B
movie_2547B
movie_2548B
movie_2549B
movie_2550B
movie_2551B
movie_2552B
movie_2553B
movie_2554B
movie_2555B
movie_2556B
movie_2557B
movie_2558B
movie_2559B
movie_2560B
movie_2561B
movie_2562B
movie_2563B
movie_2564B
movie_2565B
movie_2566B
movie_2567B
movie_2568B
movie_2569B
movie_2570B
movie_2571B
movie_2572B
movie_2573B
movie_2574B
movie_2575B
movie_2576B
movie_2577B
movie_2578B
movie_2579B
movie_2580B
movie_2581B
movie_2582B
movie_2583B
movie_2584B
movie_2585B
movie_2586B
movie_2587B
movie_2588B
movie_2589B
movie_2590B
movie_2591B
movie_2592B
movie_2593B
movie_2594B
movie_2595B
movie_2596B
movie_2597B
movie_2598B
movie_2599B
movie_2600B
movie_2601B
movie_2602B
movie_2603B
movie_2604B
movie_2605B
movie_2606B
movie_2607B
movie_2608B
movie_2609B
movie_2610B
movie_2611B
movie_2612B
movie_2613B
movie_2614B
movie_2615B
movie_2616B
movie_2617B
movie_2618B
movie_2619B
movie_2620B
movie_2621B
movie_2622B
movie_2623B
movie_2624B
movie_2625B
movie_2626B
movie_2627B
movie_2628B
movie_2629B
movie_2630B
movie_2631B
movie_2632B
movie_2633B
movie_2634B
movie_2635B
movie_2636B
movie_2637B
movie_2638B
movie_2639B
movie_2640B
movie_2641B
movie_2642B
movie_2643B
movie_2644B
movie_2645B
movie_2646B
movie_2647B
movie_2648B
movie_2649B
movie_2650B
movie_2651B
movie_2652B
movie_2653B
movie_2654B
movie_2655B
movie_2656B
movie_2657B
movie_2658B
movie_2659B
movie_2660B
movie_2661B
movie_2662B
movie_2663B
movie_2664B
movie_2665B
movie_2666B
movie_2667B
movie_2668B
movie_2669B
movie_2670B
movie_2671B
movie_2672B
movie_2673B
movie_2674B
movie_2675B
movie_2676B
movie_2677B
movie_2678B
movie_2679B
movie_2680B
movie_2681B
movie_2682B
movie_2683B
movie_2684B
movie_2685B
movie_2686B
movie_2687B
movie_2688B
movie_2689B
movie_2690B
movie_2691B
movie_2692B
movie_2693B
movie_2694B
movie_2695B
movie_2696B
movie_2697B
movie_2698B
movie_2699B
movie_2700B
movie_2701B
movie_2702B
movie_2703B
movie_2704B
movie_2705B
movie_2706B
movie_2707B
movie_2708B
movie_2709B
movie_2710B
movie_2711B
movie_2712B
movie_2713B
movie_2714B
movie_2715B
movie_2716B
movie_2717B
movie_2718B
movie_2719B
movie_2720B
movie_2721B
movie_2722B
movie_2723B
movie_2724B
movie_2725B
movie_2726B
movie_2727B
movie_2728B
movie_2729B
movie_2730B
movie_2731B
movie_2732B
movie_2733B
movie_2734B
movie_2735B
movie_2736B
movie_2737B
movie_2738B
movie_2739B
movie_2740B
movie_2741B
movie_2742B
movie_2743B
movie_2744B
movie_2745B
movie_2746B
movie_2747B
movie_2748B
movie_2749B
movie_2750B
movie_2751B
movie_2752B
movie_2753B
movie_2754B
movie_2755B
movie_2756B
movie_2757B
movie_2758B
movie_2759B
movie_2760B
movie_2761B
movie_2762B
movie_2763B
movie_2764B
movie_2765B
movie_2766B
movie_2767B
movie_2768B
movie_2769B
movie_2770B
movie_2771B
movie_2772B
movie_2773B
movie_2774B
movie_2775B
movie_2776B
movie_2777B
movie_2778B
movie_2779B
movie_2780B
movie_2781B
movie_2782B
movie_2783B
movie_2784B
movie_2785B
movie_2786B
movie_2787B
movie_2788B
movie_2789B
movie_2790B
movie_2791B
movie_2792B
movie_2793B
movie_2794B
movie_2795B
movie_2796B
movie_2797B
movie_2798B
movie_2799B
movie_2800B
movie_2801B
movie_2802B
movie_2803B
movie_2804B
movie_2805B
movie_2806B
movie_2807B
movie_2808B
movie_2809B
movie_2810B
movie_2811B
movie_2812B
movie_2813B
movie_2814B
movie_2815B
movie_2816B
movie_2817B
movie_2818B
movie_2819B
movie_2820B
movie_2821B
movie_2822B
movie_2823B
movie_2824B
movie_2825B
movie_2826B
movie_2827B
movie_2828B
movie_2829B
movie_2830B
movie_2831B
movie_2832B
movie_2833B
movie_2834B
movie_2835B
movie_2836B
movie_2837B
movie_2838B
movie_2839B
movie_2840B
movie_2841B
movie_2842B
movie_2843B
movie_2844B
movie_2845B
movie_2846B
movie_2847B
movie_2848B
movie_2849B
movie_2850B
movie_2851B
movie_2852B
movie_2853B
movie_2854B
movie_2855B
movie_2856B
movie_2857B
movie_2858B
movie_2859B
movie_2860B
movie_2861B
movie_2862B
movie_2863B
movie_2864B
movie_2865B
movie_2866B
movie_2867B
movie_2868B
movie_2869B
movie_2870B
movie_2871B
movie_2872B
movie_2873B
movie_2874B
movie_2875B
movie_2876B
movie_2877B
movie_2878B
movie_2879B
movie_2880B
movie_2881B
movie_2882B
movie_2883B
movie_2884B
movie_2885B
movie_2886B
movie_2887B
movie_2888B
movie_2889B
movie_2890B
movie_2891B
movie_2892B
movie_2893B
movie_2894B
movie_2895B
movie_2896B
movie_2897B
movie_2898B
movie_2899B
movie_2900B
movie_2901B
movie_2902B
movie_2903B
movie_2904B
movie_2905B
movie_2906B
movie_2907B
movie_2908B
movie_2909B
movie_2910B
movie_2911B
movie_2912B
movie_2913B
movie_2914B
movie_2915B
movie_2916B
movie_2917B
movie_2918B
movie_2919B
movie_2920B
movie_2921B
movie_2922B
movie_2923B
movie_2924B
movie_2925B
movie_2926B
movie_2927B
movie_2928B
movie_2929B
movie_2930B
movie_2931B
movie_2932B
movie_2933B
movie_2934B
movie_2935B
movie_2936B
movie_2937B
movie_2938B
movie_2939B
movie_2940B
movie_2941B
movie_2942B
movie_2943B
movie_2944B
movie_2945B
movie_2946B
movie_2947B
movie_2948B
movie_2949B
movie_2950B
movie_2951B
movie_2952B
movie_2953B
movie_2954B
movie_2955B
movie_2956B
movie_2957B
movie_2958B
movie_2959B
movie_2960B
movie_2961B
movie_2962B
movie_2963B
movie_2964B
movie_2965B
movie_2966B
movie_2967B
movie_2968B
movie_2969B
movie_2970B
movie_2971B
movie_2972B
movie_2973B
movie_2974B
movie_2975B
movie_2976B
movie_2977B
movie_2978B
movie_2979B
movie_2980B
movie_2981B
movie_2982B
movie_2983B
movie_2984B
movie_2985B
movie_2986B
movie_2987B
movie_2988B
movie_2989B
movie_2990B
movie_2991B
movie_2992B
movie_2993B
movie_2994B
movie_2995B
movie_2996B
movie_2997B
movie_2998B
movie_2999B
movie_3000B
movie_3001B
movie_3002B
movie_3003B
movie_3004B
movie_3005B
movie_3006B
movie_3007B
movie_3008B
movie_3009B
movie_3010B
movie_3011B
movie_3012B
movie_3013B
movie_3014B
movie_3015B
movie_3016B
movie_3017B
movie_3018B
movie_3019B
movie_3020B
movie_3021B
movie_3022B
movie_3023B
movie_3024B
movie_3025B
movie_3026B
movie_3027B
movie_3028B
movie_3029B
movie_3030B
movie_3031B
movie_3032B
movie_3033B
movie_3034B
movie_3035B
movie_3036B
movie_3037B
movie_3038B
movie_3039B
movie_3040B
movie_3041B
movie_3042B
movie_3043B
movie_3044B
movie_3045B
movie_3046B
movie_3047B
movie_3048B
movie_3049B
movie_3050B
movie_3051B
movie_3052B
movie_3053B
movie_3054B
movie_3055B
movie_3056B
movie_3057B
movie_3058B
movie_3059B
movie_3060B
movie_3061B
movie_3062B
movie_3063B
movie_3064B
movie_3065B
movie_3066B
movie_3067B
movie_3068B
movie_3069B
movie_3070B
movie_3071B
movie_3072B
movie_3073B
movie_3074B
movie_3075B
movie_3076B
movie_3077B
movie_3078B
movie_3079B
movie_3080B
movie_3081B
movie_3082B
movie_3083B
movie_3084B
movie_3085B
movie_3086B
movie_3087B
movie_3088B
movie_3089B
movie_3090B
movie_3091B
movie_3092B
movie_3093B
movie_3094B
movie_3095B
movie_3096B
movie_3097B
movie_3098B
movie_3099B
movie_3100B
movie_3101B
movie_3102B
movie_3103B
movie_3104B
movie_3105B
movie_3106B
movie_3107B
movie_3108B
movie_3109B
movie_3110B
movie_3111B
movie_3112B
movie_3113B
movie_3114B
movie_3115B
movie_3116B
movie_3117B
movie_3118B
movie_3119B
movie_3120B
movie_3121B
movie_3122B
movie_3123B
movie_3124B
movie_3125B
movie_3126B
movie_3127B
movie_3128B
movie_3129B
movie_3130B
movie_3131B
movie_3132B
movie_3133B
movie_3134B
movie_3135B
movie_3136B
movie_3137B
movie_3138B
movie_3139B
movie_3140B
movie_3141B
movie_3142B
movie_3143B
movie_3144B
movie_3145B
movie_3146B
movie_3147B
movie_3148B
movie_3149B
movie_3150B
movie_3151B
movie_3152B
movie_3153B
movie_3154B
movie_3155B
movie_3156B
movie_3157B
movie_3158B
movie_3159B
movie_3160B
movie_3161B
movie_3162B
movie_3163B
movie_3164B
movie_3165B
movie_3166B
movie_3167B
movie_3168B
movie_3169B
movie_3170B
movie_3171B
movie_3172B
movie_3173B
movie_3174B
movie_3175B
movie_3176B
movie_3177B
movie_3178B
movie_3179B
movie_3180B
movie_3181B
movie_3182B
movie_3183B
movie_3184B
movie_3185B
movie_3186B
movie_3187B
movie_3188B
movie_3189B
movie_3190B
movie_3191B
movie_3192B
movie_3193B
movie_3194B
movie_3195B
movie_3196B
movie_3197B
movie_3198B
movie_3199B
movie_3200B
movie_3201B
movie_3202B
movie_3203B
movie_3204B
movie_3205B
movie_3206B
movie_3207B
movie_3208B
movie_3209B
movie_3210B
movie_3211B
movie_3212B
movie_3213B
movie_3214B
movie_3215B
movie_3216B
movie_3217B
movie_3218B
movie_3219B
movie_3220B
movie_3221B
movie_3222B
movie_3223B
movie_3224B
movie_3225B
movie_3226B
movie_3227B
movie_3228B
movie_3229B
movie_3230B
movie_3231B
movie_3232B
movie_3233B
movie_3234B
movie_3235B
movie_3236B
movie_3237B
movie_3238B
movie_3239B
movie_3240B
movie_3241B
movie_3242B
movie_3243B
movie_3244B
movie_3245B
movie_3246B
movie_3247B
movie_3248B
movie_3249B
movie_3250B
movie_3251B
movie_3252B
movie_3253B
movie_3254B
movie_3255B
movie_3256B
movie_3257B
movie_3258B
movie_3259B
movie_3260B
movie_3261B
movie_3262B
movie_3263B
movie_3264B
movie_3265B
movie_3266B
movie_3267B
movie_3268B
movie_3269B
movie_3270B
movie_3271B
movie_3272B
movie_3273B
movie_3274B
movie_3275B
movie_3276B
movie_3277B
movie_3278B
movie_3279B
movie_3280B
movie_3281B
movie_3282B
movie_3283B
movie_3284B
movie_3285B
movie_3286B
movie_3287B
movie_3288B
movie_3289B
movie_3290B
movie_3291B
movie_3292B
movie_3293B
movie_3294B
movie_3295B
movie_3296B
movie_3297B
movie_3298B
movie_3299B
movie_3300B
movie_3301B
movie_3302B
movie_3303B
movie_3304B
movie_3305B
movie_3306B
movie_3307B
movie_3308B
movie_3309B
movie_3310B
movie_3311B
movie_3312B
movie_3313B
movie_3314B
movie_3315B
movie_3316B
movie_3317B
movie_3318B
movie_3319B
movie_3320B
movie_3321B
movie_3322B
movie_3323B
movie_3324B
movie_3325B
movie_3326B
movie_3327B
movie_3328B
movie_3329B
movie_3330B
movie_3331B
movie_3332B
movie_3333B
movie_3334B
movie_3335B
movie_3336B
movie_3337B
movie_3338B
movie_3339B
movie_3340B
movie_3341B
movie_3342B
movie_3343B
movie_3344B
movie_3345B
movie_3346B
movie_3347B
movie_3348B
movie_3349B
movie_3350B
movie_3351B
movie_3352B
movie_3353B
movie_3354B
movie_3355B
movie_3356B
movie_3357B
movie_3358B
movie_3359B
movie_3360B
movie_3361B
movie_3362B
movie_3363B
movie_3364B
movie_3365B
movie_3366B
movie_3367B
movie_3368B
movie_3369B
movie_3370B
movie_3371B
movie_3372B
movie_3373B
movie_3374B
movie_3375B
movie_3376B
movie_3377B
movie_3378B
movie_3379B
movie_3380B
movie_3381B
movie_3382B
movie_3383B
movie_3384B
movie_3385B
movie_3386B
movie_3387B
movie_3388B
movie_3389B
movie_3390B
movie_3391B
movie_3392B
movie_3393B
movie_3394B
movie_3395B
movie_3396B
movie_3397B
movie_3398B
movie_3399B
movie_3400B
movie_3401B
movie_3402B
movie_3403B
movie_3404B
movie_3405B
movie_3406B
movie_3407B
movie_3408B
movie_3409B
movie_3410B
movie_3411B
movie_3412B
movie_3413B
movie_3414B
movie_3415B
movie_3416B
movie_3417B
movie_3418B
movie_3419B
movie_3420B
movie_3421B
movie_3422B
movie_3423B
movie_3424B
movie_3425B
movie_3426B
movie_3427B
movie_3428B
movie_3429B
movie_3430B
movie_3431B
movie_3432B
movie_3433B
movie_3434B
movie_3435B
movie_3436B
movie_3437B
movie_3438B
movie_3439B
movie_3440B
movie_3441B
movie_3442B
movie_3443B
movie_3444B
movie_3445B
movie_3446B
movie_3447B
movie_3448B
movie_3449B
movie_3450B
movie_3451B
movie_3452B
movie_3453B
movie_3454B
movie_3455B
movie_3456B
movie_3457B
movie_3458B
movie_3459B
movie_3460B
movie_3461B
movie_3462B
movie_3463B
movie_3464B
movie_3465B
movie_3466B
movie_3467B
movie_3468B
movie_3469B
movie_3470B
movie_3471B
movie_3472B
movie_3473B
movie_3474B
movie_3475B
movie_3476B
movie_3477B
movie_3478B
movie_3479B
movie_3480B
movie_3481B
movie_3482B
movie_3483B
movie_3484B
movie_3485B
movie_3486B
movie_3487B
movie_3488B
movie_3489B
movie_3490B
movie_3491B
movie_3492B
movie_3493B
movie_3494B
movie_3495B
movie_3496B
movie_3497B
movie_3498B
movie_3499B
movie_3500B
movie_3501B
movie_3502B
movie_3503B
movie_3504B
movie_3505B
movie_3506B
movie_3507B
movie_3508B
movie_3509B
movie_3510B
movie_3511B
movie_3512B
movie_3513B
movie_3514B
movie_3515B
movie_3516B
movie_3517B
movie_3518B
movie_3519B
movie_3520B
movie_3521B
movie_3522B
movie_3523B
movie_3524B
movie_3525B
movie_3526B
movie_3527B
movie_3528B
movie_3529B
movie_3530B
movie_3531B
movie_3532B
movie_3533B
movie_3534B
movie_3535B
movie_3536B
movie_3537B
movie_3538B
movie_3539B
movie_3540B
movie_3541B
movie_3542B
movie_3543B
movie_3544B
movie_3545B
movie_3546B
movie_3547B
movie_3548B
movie_3549B
movie_3550B
movie_3551B
movie_3552B
movie_3553B
movie_3554B
movie_3555B
movie_3556B
movie_3557B
movie_3558B
movie_3559B
movie_3560B
movie_3561B
movie_3562B
movie_3563B
movie_3564B
movie_3565B
movie_3566B
movie_3567B
movie_3568B
movie_3569B
movie_3570B
movie_3571B
movie_3572B
movie_3573B
movie_3574B
movie_3575B
movie_3576B
movie_3577B
movie_3578B
movie_3579B
movie_3580B
movie_3581B
movie_3582B
movie_3583B
movie_3584B
movie_3585B
movie_3586B
movie_3587B
movie_3588B
movie_3589B
movie_3590B
movie_3591B
movie_3592B
movie_3593B
movie_3594B
movie_3595B
movie_3596B
movie_3597B
movie_3598B
movie_3599B
movie_3600B
movie_3601B
movie_3602B
movie_3603B
movie_3604B
movie_3605B
movie_3606B
movie_3607B
movie_3608B
movie_3609B
movie_3610B
movie_3611B
movie_3612B
movie_3613B
movie_3614B
movie_3615B
movie_3616B
movie_3617B
movie_3618B
movie_3619B
movie_3620B
movie_3621B
movie_3622B
movie_3623B
movie_3624B
movie_3625B
movie_3626B
movie_3627B
movie_3628B
movie_3629B
movie_3630B
movie_3631B
movie_3632B
movie_3633B
movie_3634B
movie_3635B
movie_3636B
movie_3637B
movie_3638B
movie_3639B
movie_3640B
movie_3641B
movie_3642B
movie_3643B
movie_3644B
movie_3645B
movie_3646B
movie_3647B
movie_3648B
movie_3649B
movie_3650B
movie_3651B
movie_3652B
movie_3653B
movie_3654B
movie_3655B
movie_3656B
movie_3657B
movie_3658B
movie_3659B
movie_3660B
movie_3661B
movie_3662B
movie_3663B
movie_3664B
movie_3665B
movie_3666B
movie_3667B
movie_3668B
movie_3669B
movie_3670B
movie_3671B
movie_3672B
movie_3673B
movie_3674B
movie_3675B
movie_3676B
movie_3677B
movie_3678B
movie_3679B
movie_3680B
movie_3681B
movie_3682B
movie_3683B
movie_3684B
movie_3685B
movie_3686B
movie_3687B
movie_3688B
movie_3689B
movie_3690B
movie_3691B
movie_3692B
movie_3693B
movie_3694B
movie_3695B
movie_3696B
movie_3697B
movie_3698B
movie_3699B
movie_3700B
movie_3701B
movie_3702B
movie_3703B
movie_3704B
movie_3705B
movie_3706B
movie_3707B
movie_3708B
movie_3709B
movie_3710B
movie_3711B
movie_3712B
movie_3713B
movie_3714B
movie_3715B
movie_3716B
movie_3717B
movie_3718B
movie_3719B
movie_3720B
movie_3721B
movie_3722B
movie_3723B
movie_3724B
movie_3725B
movie_3726B
movie_3727B
movie_3728B
movie_3729B
movie_3730B
movie_3731B
movie_3732B
movie_3733B
movie_3734B
movie_3735B
movie_3736B
movie_3737B
movie_3738B
movie_3739B
movie_3740B
movie_3741B
movie_3742B
movie_3743B
movie_3744B
movie_3745B
movie_3746B
movie_3747B
movie_3748B
movie_3749B
movie_3750B
movie_3751B
movie_3752B
movie_3753B
movie_3754B
movie_3755B
movie_3756B
movie_3757B
movie_3758B
movie_3759B
movie_3760B
movie_3761B
movie_3762B
movie_3763B
movie_3764B
movie_3765B
movie_3766B
movie_3767B
movie_3768B
movie_3769B
movie_3770B
movie_3771B
movie_3772B
movie_3773B
movie_3774B
movie_3775B
movie_3776B
movie_3777B
movie_3778B
movie_3779B
movie_3780B
movie_3781B
movie_3782B
movie_3783B
movie_3784B
movie_3785B
movie_3786B
movie_3787B
movie_3788B
movie_3789B
movie_3790B
movie_3791B
movie_3792B
movie_3793B
movie_3794B
movie_3795B
movie_3796B
movie_3797B
movie_3798B
movie_3799B
movie_3800B
movie_3801B
movie_3802B
movie_3803B
movie_3804B
movie_3805B
movie_3806B
movie_3807B
movie_3808B
movie_3809B
movie_3810B
movie_3811B
movie_3812B
movie_3813B
movie_3814B
movie_3816B
movie_3817B
movie_3818B
movie_3819B
movie_3820B
movie_3821B
movie_3822B
movie_3823B
movie_3824B
movie_3825B
movie_3826B
movie_3827B
movie_3828B
movie_3829B
movie_3830B
movie_3831B
movie_3832B
movie_3833B
movie_3834B
movie_3835B
movie_3836B
movie_3837B
movie_3838B
movie_3839B
movie_3840B
movie_3841B
movie_3842B
movie_3843B
movie_3844B
movie_3845B
movie_3846B
movie_3847B
movie_3848B
movie_3849B
movie_3850B
movie_3851B
movie_3852B
movie_3853B
movie_3854B
movie_3855B
movie_3856B
movie_3857B
movie_3858B
movie_3859B
movie_3860B
movie_3861B
movie_3862B
movie_3863B
movie_3864B
movie_3865B
movie_3866B
movie_3867B
movie_3868B
movie_3869B
movie_3870B
movie_3871B
movie_3872B
movie_3873B
movie_3874B
movie_3875B
movie_3876B
movie_3877B
movie_3878B
movie_3879B
movie_3880B
movie_3881B
movie_3882B
movie_3883B
movie_3884B
movie_3885B
movie_3886B
movie_3887B
movie_3888B
movie_3889B
movie_3890B
movie_3891B
movie_3892B
movie_3893B
movie_3894B
movie_3895B
movie_3896B
movie_3897B
movie_3898B
movie_3899B
movie_3900B
movie_3901B
movie_3902B
movie_3903B
movie_3904B
movie_3905B
movie_3906B
movie_3907B
movie_3908B
movie_3909B
movie_3910B
movie_3911B
movie_3912B
movie_3913B
movie_3914B
movie_3915B
movie_3916B
movie_3917B
movie_3918B
movie_3919B
movie_3920B
movie_3921B
movie_3922B
movie_3923B
movie_3924B
movie_3925B
movie_3926B
movie_3927B
movie_3928B
movie_3929B
movie_3930B
movie_3931B
movie_3932B
movie_3933B
movie_3934B
movie_3935B
movie_3936B
movie_3937B
movie_3938B
movie_3939B
movie_3940B
movie_3941B
movie_3942B
movie_3943B
movie_3944B
movie_3945B
movie_3946B
movie_3947B
movie_3948B
movie_3949B
movie_3950B
movie_3951B
movie_3952
??
Const_6Const*
_output_shapes	
:?*
dtype0	*??
value??B??	?"??                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      
T
Const_7Const*
_output_shapes
:*
dtype0*
valueBBFBM
`
Const_8Const*
_output_shapes
:*
dtype0	*%
valueB	"               
?
Const_9Const*
_output_shapes
:*
dtype0*X
valueOBMBgroup_1Bgroup_56Bgroup_25Bgroup_45Bgroup_50Bgroup_35Bgroup_18
?
Const_10Const*
_output_shapes
:*
dtype0	*M
valueDBB	"8                                                  
?
Const_11Const*
_output_shapes
:*
dtype0*?
value?B?Boccupation_10Boccupation_16Boccupation_15Boccupation_7Boccupation_20Boccupation_9Boccupation_1Boccupation_12Boccupation_17Boccupation_0Boccupation_3Boccupation_14Boccupation_4Boccupation_11Boccupation_8Boccupation_19Boccupation_2Boccupation_18Boccupation_5Boccupation_13Boccupation_6
?
Const_12Const*
_output_shapes
:*
dtype0	*?
value?B?	"?                                                                	       
                                                                             
?
StatefulPartitionedCallStatefulPartitionedCallhash_table_3Const_5Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_26009
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_2Const_7Const_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_26017
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_1Const_9Const_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_26025
?
StatefulPartitionedCall_3StatefulPartitionedCall
hash_tableConst_11Const_12*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *#
fR
__inference_<lambda>_26033
z
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3
??
Const_13Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer_with_weights-3
,layer-43
-layer-44
.layer-45
/layer_with_weights-4
/layer-46
0layer-47
1layer_with_weights-5
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer_with_weights-6
:layer-57
;layer_with_weights-7
;layer-58
<layer_with_weights-8
<layer-59
=layer_with_weights-9
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer_with_weights-10
Blayer-65
Clayer_with_weights-11
Clayer-66
Dlayer-67
Elayer-68
Flayer_with_weights-12
Flayer-69
Glayer_with_weights-13
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-14
Klayer-74
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
R_default_save_signature
S	optimizer
T
signatures*
* 
9
U	keras_api
Vinput_vocabulary
Wlookup_table* 
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^
embeddings*
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e
embeddings*
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias*
* 
* 

t	keras_api* 

u	keras_api* 
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	keras_api* 
?
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 

?	keras_api* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
* 
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
<
?	keras_api
?input_vocabulary
?lookup_table* 
<
?	keras_api
?input_vocabulary
?lookup_table* 
<
?	keras_api
?input_vocabulary
?lookup_table* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
^0
e1
r2
s3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34*
?
^0
r1
s2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
R_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter

?decay
?learning_rate^accumulator?raccumulator?saccumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator?*

?serving_default* 
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 

^0*

^0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
nh
VARIABLE_VALUEmovie_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

e0*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
lf
VARIABLE_VALUEgenres_vector/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

r0
s1*

r0
s1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
zt
VARIABLE_VALUE*process_movie_embedding_with_genres/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE(process_movie_embedding_with_genres/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
D
?0
?1
?2
?3
?4
?5
?6
?7*
D
?0
?1
?2
?3
?4
?5
?6
?7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
lf
VARIABLE_VALUEsex_embedding/embeddings:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
rl
VARIABLE_VALUEage_group_embedding/embeddings:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

?0*

?0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
sm
VARIABLE_VALUEoccupation_embedding/embeddings:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ic
VARIABLE_VALUEbatch_normalization/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEbatch_normalization/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEbatch_normalization/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#batch_normalization/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_2/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_1/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_1/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_1/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_1/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmulti_head_attention/key/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmulti_head_attention/value/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention/attention_output/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*multi_head_attention/attention_output/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
+
e0
?1
?2
?3
?4*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74*

?0
?1*
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
OI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

e0*
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
4
?0
?1
?2
?3
?4
?5*
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
* 

?trace_0* 

?trace_0* 

?trace_0* 
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

?0
?1*
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

?0
?1*
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
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
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

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUE.Adagrad/movie_embedding/embeddings/accumulator`layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adagrad/process_movie_embedding_with_genres/kernel/accumulator\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adagrad/process_movie_embedding_with_genres/bias/accumulatorZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adagrad/layer_normalization/gamma/accumulator[layer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adagrad/layer_normalization/beta/accumulatorZlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense/kernel/accumulator\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdagrad/dense/bias/accumulatorZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adagrad/sex_embedding/embeddings/accumulator`layer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2Adagrad/age_group_embedding/embeddings/accumulator`layer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3Adagrad/occupation_embedding/embeddings/accumulator`layer_with_weights-8/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE/Adagrad/layer_normalization_1/gamma/accumulator[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adagrad/layer_normalization_1/beta/accumulatorZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adagrad/dense_1/kernel/accumulator]layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense_1/bias/accumulator[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE-Adagrad/batch_normalization/gamma/accumulator\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adagrad/batch_normalization/beta/accumulator[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adagrad/dense_2/kernel/accumulator]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense_2/bias/accumulator[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE/Adagrad/batch_normalization_1/gamma/accumulator\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adagrad/batch_normalization_1/beta/accumulator[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adagrad/dense_3/kernel/accumulator]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adagrad/dense_3/bias/accumulator[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adagrad/multi_head_attention/query/kernel/accumulatorLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3Adagrad/multi_head_attention/query/bias/accumulatorLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3Adagrad/multi_head_attention/key/kernel/accumulatorLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE1Adagrad/multi_head_attention/key/bias/accumulatorLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adagrad/multi_head_attention/value/kernel/accumulatorLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3Adagrad/multi_head_attention/value/bias/accumulatorLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE@Adagrad/multi_head_attention/attention_output/kernel/accumulatorMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adagrad/multi_head_attention/attention_output/bias/accumulatorMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_age_groupPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_occupationPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
"serving_default_sequence_movie_idsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_sequence_ratingsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_sexPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_target_movie_idPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_user_idPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_4StatefulPartitionedCallserving_default_age_groupserving_default_occupation"serving_default_sequence_movie_ids serving_default_sequence_ratingsserving_default_sexserving_default_target_movie_idserving_default_user_idhash_table_3Constmovie_embedding/embeddingsgenres_vector/embeddings*process_movie_embedding_with_genres/kernel(process_movie_embedding_with_genres/biasConst_1!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization/gammalayer_normalization/betadense/kernel
dense/bias
hash_tableConst_2hash_table_1Const_3hash_table_2Const_4sex_embedding/embeddingsage_group_embedding/embeddingsoccupation_embedding/embeddingslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense_2/kerneldense_2/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betadense_3/kerneldense_3/bias*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*E
_read_only_resource_inputs'
%#	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_23730
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filename.movie_embedding/embeddings/Read/ReadVariableOp,genres_vector/embeddings/Read/ReadVariableOp>process_movie_embedding_with_genres/kernel/Read/ReadVariableOp<process_movie_embedding_with_genres/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp,sex_embedding/embeddings/Read/ReadVariableOp2age_group_embedding/embeddings/Read/ReadVariableOp3occupation_embedding/embeddings/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpBAdagrad/movie_embedding/embeddings/accumulator/Read/ReadVariableOpRAdagrad/process_movie_embedding_with_genres/kernel/accumulator/Read/ReadVariableOpPAdagrad/process_movie_embedding_with_genres/bias/accumulator/Read/ReadVariableOpAAdagrad/layer_normalization/gamma/accumulator/Read/ReadVariableOp@Adagrad/layer_normalization/beta/accumulator/Read/ReadVariableOp4Adagrad/dense/kernel/accumulator/Read/ReadVariableOp2Adagrad/dense/bias/accumulator/Read/ReadVariableOp@Adagrad/sex_embedding/embeddings/accumulator/Read/ReadVariableOpFAdagrad/age_group_embedding/embeddings/accumulator/Read/ReadVariableOpGAdagrad/occupation_embedding/embeddings/accumulator/Read/ReadVariableOpCAdagrad/layer_normalization_1/gamma/accumulator/Read/ReadVariableOpBAdagrad/layer_normalization_1/beta/accumulator/Read/ReadVariableOp6Adagrad/dense_1/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_1/bias/accumulator/Read/ReadVariableOpAAdagrad/batch_normalization/gamma/accumulator/Read/ReadVariableOp@Adagrad/batch_normalization/beta/accumulator/Read/ReadVariableOp6Adagrad/dense_2/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_2/bias/accumulator/Read/ReadVariableOpCAdagrad/batch_normalization_1/gamma/accumulator/Read/ReadVariableOpBAdagrad/batch_normalization_1/beta/accumulator/Read/ReadVariableOp6Adagrad/dense_3/kernel/accumulator/Read/ReadVariableOp4Adagrad/dense_3/bias/accumulator/Read/ReadVariableOpIAdagrad/multi_head_attention/query/kernel/accumulator/Read/ReadVariableOpGAdagrad/multi_head_attention/query/bias/accumulator/Read/ReadVariableOpGAdagrad/multi_head_attention/key/kernel/accumulator/Read/ReadVariableOpEAdagrad/multi_head_attention/key/bias/accumulator/Read/ReadVariableOpIAdagrad/multi_head_attention/value/kernel/accumulator/Read/ReadVariableOpGAdagrad/multi_head_attention/value/bias/accumulator/Read/ReadVariableOpTAdagrad/multi_head_attention/attention_output/kernel/accumulator/Read/ReadVariableOpRAdagrad/multi_head_attention/attention_output/bias/accumulator/Read/ReadVariableOpConst_13*U
TinN
L2J	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_26299
?
StatefulPartitionedCall_6StatefulPartitionedCallsaver_filenamemovie_embedding/embeddingsgenres_vector/embeddings*process_movie_embedding_with_genres/kernel(process_movie_embedding_with_genres/biaslayer_normalization/gammalayer_normalization/betadense/kernel
dense/biassex_embedding/embeddingsage_group_embedding/embeddingsoccupation_embedding/embeddingslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_2/kerneldense_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_3/kerneldense_3/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasAdagrad/iterAdagrad/decayAdagrad/learning_ratetotal_1count_1totalcount.Adagrad/movie_embedding/embeddings/accumulator>Adagrad/process_movie_embedding_with_genres/kernel/accumulator<Adagrad/process_movie_embedding_with_genres/bias/accumulator-Adagrad/layer_normalization/gamma/accumulator,Adagrad/layer_normalization/beta/accumulator Adagrad/dense/kernel/accumulatorAdagrad/dense/bias/accumulator,Adagrad/sex_embedding/embeddings/accumulator2Adagrad/age_group_embedding/embeddings/accumulator3Adagrad/occupation_embedding/embeddings/accumulator/Adagrad/layer_normalization_1/gamma/accumulator.Adagrad/layer_normalization_1/beta/accumulator"Adagrad/dense_1/kernel/accumulator Adagrad/dense_1/bias/accumulator-Adagrad/batch_normalization/gamma/accumulator,Adagrad/batch_normalization/beta/accumulator"Adagrad/dense_2/kernel/accumulator Adagrad/dense_2/bias/accumulator/Adagrad/batch_normalization_1/gamma/accumulator.Adagrad/batch_normalization_1/beta/accumulator"Adagrad/dense_3/kernel/accumulator Adagrad/dense_3/bias/accumulator5Adagrad/multi_head_attention/query/kernel/accumulator3Adagrad/multi_head_attention/query/bias/accumulator3Adagrad/multi_head_attention/key/kernel/accumulator1Adagrad/multi_head_attention/key/bias/accumulator5Adagrad/multi_head_attention/value/kernel/accumulator3Adagrad/multi_head_attention/value/bias/accumulator@Adagrad/multi_head_attention/attention_output/kernel/accumulator>Adagrad/multi_head_attention/attention_output/bias/accumulator*T
TinM
K2I*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_26525??*
?
?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291

inputs	)
embedding_lookup_21285:	?>
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21285inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21285*+
_output_shapes
:?????????>*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21285*+
_output_shapes
:?????????>?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????>Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_25819

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21240p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_25647

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_21923p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_3_layer_call_fn_25893

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_22129p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24978

inputs	)
embedding_lookup_24972:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_24972inputs*
Tindices0	*)
_class
loc:@embedding_lookup/24972*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/24972*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_25883

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???>`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_260255
1key_value_init54_lookuptableimportv2_table_handle-
)key_value_init54_lookuptableimportv2_keys/
+key_value_init54_lookuptableimportv2_values	
identity??$key_value_init54/LookupTableImportV2?
$key_value_init54/LookupTableImportV2LookupTableImportV21key_value_init54_lookuptableimportv2_table_handle)key_value_init54_lookuptableimportv2_keys+key_value_init54_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init54/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init54/LookupTableImportV2$key_value_init54/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
,
__inference__destroyer_26001
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?$
?
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876

inputs+
mul_3_readvariableop_resource:>)
add_readvariableop_resource:>
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????>L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????>n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:>*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:>*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????>r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_25683

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21158p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_259965
1key_value_init97_lookuptableimportv2_table_handle-
)key_value_init97_lookuptableimportv2_keys/
+key_value_init97_lookuptableimportv2_values	
identity??$key_value_init97/LookupTableImportV2?
$key_value_init97/LookupTableImportV2LookupTableImportV21key_value_init97_lookuptableimportv2_table_handle)key_value_init97_lookuptableimportv2_keys+key_value_init97_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init97/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init97/LookupTableImportV2$key_value_init97/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
-__inference_genres_vector_layer_call_fn_24969

inputs	
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
@__inference_add_1_layer_call_and_return_conditional_losses_21817

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_25466

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?	
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_22129

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_25449

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_22275s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?%
__inference__traced_save_26299
file_prefix9
5savev2_movie_embedding_embeddings_read_readvariableop7
3savev2_genres_vector_embeddings_read_readvariableopI
Esavev2_process_movie_embedding_with_genres_kernel_read_readvariableopG
Csavev2_process_movie_embedding_with_genres_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop7
3savev2_sex_embedding_embeddings_read_readvariableop=
9savev2_age_group_embedding_embeddings_read_readvariableop>
:savev2_occupation_embedding_embeddings_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableop+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopM
Isavev2_adagrad_movie_embedding_embeddings_accumulator_read_readvariableop]
Ysavev2_adagrad_process_movie_embedding_with_genres_kernel_accumulator_read_readvariableop[
Wsavev2_adagrad_process_movie_embedding_with_genres_bias_accumulator_read_readvariableopL
Hsavev2_adagrad_layer_normalization_gamma_accumulator_read_readvariableopK
Gsavev2_adagrad_layer_normalization_beta_accumulator_read_readvariableop?
;savev2_adagrad_dense_kernel_accumulator_read_readvariableop=
9savev2_adagrad_dense_bias_accumulator_read_readvariableopK
Gsavev2_adagrad_sex_embedding_embeddings_accumulator_read_readvariableopQ
Msavev2_adagrad_age_group_embedding_embeddings_accumulator_read_readvariableopR
Nsavev2_adagrad_occupation_embedding_embeddings_accumulator_read_readvariableopN
Jsavev2_adagrad_layer_normalization_1_gamma_accumulator_read_readvariableopM
Isavev2_adagrad_layer_normalization_1_beta_accumulator_read_readvariableopA
=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_1_bias_accumulator_read_readvariableopL
Hsavev2_adagrad_batch_normalization_gamma_accumulator_read_readvariableopK
Gsavev2_adagrad_batch_normalization_beta_accumulator_read_readvariableopA
=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_2_bias_accumulator_read_readvariableopN
Jsavev2_adagrad_batch_normalization_1_gamma_accumulator_read_readvariableopM
Isavev2_adagrad_batch_normalization_1_beta_accumulator_read_readvariableopA
=savev2_adagrad_dense_3_kernel_accumulator_read_readvariableop?
;savev2_adagrad_dense_3_bias_accumulator_read_readvariableopT
Psavev2_adagrad_multi_head_attention_query_kernel_accumulator_read_readvariableopR
Nsavev2_adagrad_multi_head_attention_query_bias_accumulator_read_readvariableopR
Nsavev2_adagrad_multi_head_attention_key_kernel_accumulator_read_readvariableopP
Lsavev2_adagrad_multi_head_attention_key_bias_accumulator_read_readvariableopT
Psavev2_adagrad_multi_head_attention_value_kernel_accumulator_read_readvariableopR
Nsavev2_adagrad_multi_head_attention_value_bias_accumulator_read_readvariableop_
[savev2_adagrad_multi_head_attention_attention_output_kernel_accumulator_read_readvariableop]
Ysavev2_adagrad_multi_head_attention_attention_output_bias_accumulator_read_readvariableop
savev2_const_13

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?&
value?&B?&IB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-8/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_movie_embedding_embeddings_read_readvariableop3savev2_genres_vector_embeddings_read_readvariableopEsavev2_process_movie_embedding_with_genres_kernel_read_readvariableopCsavev2_process_movie_embedding_with_genres_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop3savev2_sex_embedding_embeddings_read_readvariableop9savev2_age_group_embedding_embeddings_read_readvariableop:savev2_occupation_embedding_embeddings_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopIsavev2_adagrad_movie_embedding_embeddings_accumulator_read_readvariableopYsavev2_adagrad_process_movie_embedding_with_genres_kernel_accumulator_read_readvariableopWsavev2_adagrad_process_movie_embedding_with_genres_bias_accumulator_read_readvariableopHsavev2_adagrad_layer_normalization_gamma_accumulator_read_readvariableopGsavev2_adagrad_layer_normalization_beta_accumulator_read_readvariableop;savev2_adagrad_dense_kernel_accumulator_read_readvariableop9savev2_adagrad_dense_bias_accumulator_read_readvariableopGsavev2_adagrad_sex_embedding_embeddings_accumulator_read_readvariableopMsavev2_adagrad_age_group_embedding_embeddings_accumulator_read_readvariableopNsavev2_adagrad_occupation_embedding_embeddings_accumulator_read_readvariableopJsavev2_adagrad_layer_normalization_1_gamma_accumulator_read_readvariableopIsavev2_adagrad_layer_normalization_1_beta_accumulator_read_readvariableop=savev2_adagrad_dense_1_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_1_bias_accumulator_read_readvariableopHsavev2_adagrad_batch_normalization_gamma_accumulator_read_readvariableopGsavev2_adagrad_batch_normalization_beta_accumulator_read_readvariableop=savev2_adagrad_dense_2_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_2_bias_accumulator_read_readvariableopJsavev2_adagrad_batch_normalization_1_gamma_accumulator_read_readvariableopIsavev2_adagrad_batch_normalization_1_beta_accumulator_read_readvariableop=savev2_adagrad_dense_3_kernel_accumulator_read_readvariableop;savev2_adagrad_dense_3_bias_accumulator_read_readvariableopPsavev2_adagrad_multi_head_attention_query_kernel_accumulator_read_readvariableopNsavev2_adagrad_multi_head_attention_query_bias_accumulator_read_readvariableopNsavev2_adagrad_multi_head_attention_key_kernel_accumulator_read_readvariableopLsavev2_adagrad_multi_head_attention_key_bias_accumulator_read_readvariableopPsavev2_adagrad_multi_head_attention_value_kernel_accumulator_read_readvariableopNsavev2_adagrad_multi_head_attention_value_bias_accumulator_read_readvariableop[savev2_adagrad_multi_head_attention_attention_output_kernel_accumulator_read_readvariableopYsavev2_adagrad_multi_head_attention_attention_output_bias_accumulator_read_readvariableopsavev2_const_13"/device:CPU:0*
_output_shapes
 *W
dtypesM
K2I	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?>:	?:P>:>:>:>:>>:>::::>:>:
??:?:?:?:?:?:
??:?:?:?:?:?:	?::>>:>:>>:>:>>:>:>>:>: : : : : : : :	?>:P>:>:>:>:>>:>::::>:>:
??:?:?:?:
??:?:?:?:	?::>>:>:>>:>:>>:>:>>:>: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?>:%!

_output_shapes
:	?:$ 

_output_shapes

:P>: 

_output_shapes
:>: 

_output_shapes
:>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$	 

_output_shapes

::$
 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:>: 

_output_shapes
:>:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::($
"
_output_shapes
:>>:$ 

_output_shapes

:>:($
"
_output_shapes
:>>:$ 

_output_shapes

:>:( $
"
_output_shapes
:>>:$! 

_output_shapes

:>:("$
"
_output_shapes
:>>: #

_output_shapes
:>:$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :%+!

_output_shapes
:	?>:$, 

_output_shapes

:P>: -

_output_shapes
:>: .

_output_shapes
:>: /

_output_shapes
:>:$0 

_output_shapes

:>>: 1

_output_shapes
:>:$2 

_output_shapes

::$3 

_output_shapes

::$4 

_output_shapes

:: 5

_output_shapes
:>: 6

_output_shapes
:>:&7"
 
_output_shapes
:
??:!8

_output_shapes	
:?:!9

_output_shapes	
:?:!:

_output_shapes	
:?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:!=

_output_shapes	
:?:!>

_output_shapes	
:?:%?!

_output_shapes
:	?: @

_output_shapes
::(A$
"
_output_shapes
:>>:$B 

_output_shapes

:>:(C$
"
_output_shapes
:>>:$D 

_output_shapes

:>:(E$
"
_output_shapes
:>>:$F 

_output_shapes

:>:(G$
"
_output_shapes
:>>: H

_output_shapes
:>:I

_output_shapes
: 
?
C
'__inference_dropout_layer_call_fn_25300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_21623d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
~
F__inference_concatenate_layer_call_and_return_conditional_losses_21827

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????:?????????:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_23091
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_id
unknown
	unknown_0	
	unknown_1:	?>
	unknown_2:	?
	unknown_3:P>
	unknown_4:>
	unknown_5
	unknown_6:>>
	unknown_7:>
	unknown_8:>>
	unknown_9:> 

unknown_10:>>

unknown_11:> 

unknown_12:>>

unknown_13:>

unknown_14:>

unknown_15:>

unknown_16:>>

unknown_17:>

unknown_18

unknown_19	

unknown_20

unknown_21	

unknown_22

unknown_23	

unknown_24:

unknown_25:

unknown_26:

unknown_27:>

unknown_28:>

unknown_29:
??

unknown_30:	?

unknown_31:	?

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*A
_read_only_resource_inputs#
!	
 !"#$%&)*+,/012*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_22901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?$
?
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25582

inputs+
mul_3_readvariableop_resource:>)
add_readvariableop_resource:>
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????>L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????>n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:>*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:>*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????>r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_25454

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?4
!__inference__traced_restore_26525
file_prefix>
+assignvariableop_movie_embedding_embeddings:	?>>
+assignvariableop_1_genres_vector_embeddings:	?O
=assignvariableop_2_process_movie_embedding_with_genres_kernel:P>I
;assignvariableop_3_process_movie_embedding_with_genres_bias:>:
,assignvariableop_4_layer_normalization_gamma:>9
+assignvariableop_5_layer_normalization_beta:>1
assignvariableop_6_dense_kernel:>>+
assignvariableop_7_dense_bias:>=
+assignvariableop_8_sex_embedding_embeddings:C
1assignvariableop_9_age_group_embedding_embeddings:E
3assignvariableop_10_occupation_embedding_embeddings:=
/assignvariableop_11_layer_normalization_1_gamma:><
.assignvariableop_12_layer_normalization_1_beta:>6
"assignvariableop_13_dense_1_kernel:
??/
 assignvariableop_14_dense_1_bias:	?<
-assignvariableop_15_batch_normalization_gamma:	?;
,assignvariableop_16_batch_normalization_beta:	?B
3assignvariableop_17_batch_normalization_moving_mean:	?F
7assignvariableop_18_batch_normalization_moving_variance:	?6
"assignvariableop_19_dense_2_kernel:
??/
 assignvariableop_20_dense_2_bias:	?>
/assignvariableop_21_batch_normalization_1_gamma:	?=
.assignvariableop_22_batch_normalization_1_beta:	?D
5assignvariableop_23_batch_normalization_1_moving_mean:	?H
9assignvariableop_24_batch_normalization_1_moving_variance:	?5
"assignvariableop_25_dense_3_kernel:	?.
 assignvariableop_26_dense_3_bias:K
5assignvariableop_27_multi_head_attention_query_kernel:>>E
3assignvariableop_28_multi_head_attention_query_bias:>I
3assignvariableop_29_multi_head_attention_key_kernel:>>C
1assignvariableop_30_multi_head_attention_key_bias:>K
5assignvariableop_31_multi_head_attention_value_kernel:>>E
3assignvariableop_32_multi_head_attention_value_bias:>V
@assignvariableop_33_multi_head_attention_attention_output_kernel:>>L
>assignvariableop_34_multi_head_attention_attention_output_bias:>*
 assignvariableop_35_adagrad_iter:	 +
!assignvariableop_36_adagrad_decay: 3
)assignvariableop_37_adagrad_learning_rate: %
assignvariableop_38_total_1: %
assignvariableop_39_count_1: #
assignvariableop_40_total: #
assignvariableop_41_count: U
Bassignvariableop_42_adagrad_movie_embedding_embeddings_accumulator:	?>d
Rassignvariableop_43_adagrad_process_movie_embedding_with_genres_kernel_accumulator:P>^
Passignvariableop_44_adagrad_process_movie_embedding_with_genres_bias_accumulator:>O
Aassignvariableop_45_adagrad_layer_normalization_gamma_accumulator:>N
@assignvariableop_46_adagrad_layer_normalization_beta_accumulator:>F
4assignvariableop_47_adagrad_dense_kernel_accumulator:>>@
2assignvariableop_48_adagrad_dense_bias_accumulator:>R
@assignvariableop_49_adagrad_sex_embedding_embeddings_accumulator:X
Fassignvariableop_50_adagrad_age_group_embedding_embeddings_accumulator:Y
Gassignvariableop_51_adagrad_occupation_embedding_embeddings_accumulator:Q
Cassignvariableop_52_adagrad_layer_normalization_1_gamma_accumulator:>P
Bassignvariableop_53_adagrad_layer_normalization_1_beta_accumulator:>J
6assignvariableop_54_adagrad_dense_1_kernel_accumulator:
??C
4assignvariableop_55_adagrad_dense_1_bias_accumulator:	?P
Aassignvariableop_56_adagrad_batch_normalization_gamma_accumulator:	?O
@assignvariableop_57_adagrad_batch_normalization_beta_accumulator:	?J
6assignvariableop_58_adagrad_dense_2_kernel_accumulator:
??C
4assignvariableop_59_adagrad_dense_2_bias_accumulator:	?R
Cassignvariableop_60_adagrad_batch_normalization_1_gamma_accumulator:	?Q
Bassignvariableop_61_adagrad_batch_normalization_1_beta_accumulator:	?I
6assignvariableop_62_adagrad_dense_3_kernel_accumulator:	?B
4assignvariableop_63_adagrad_dense_3_bias_accumulator:_
Iassignvariableop_64_adagrad_multi_head_attention_query_kernel_accumulator:>>Y
Gassignvariableop_65_adagrad_multi_head_attention_query_bias_accumulator:>]
Gassignvariableop_66_adagrad_multi_head_attention_key_kernel_accumulator:>>W
Eassignvariableop_67_adagrad_multi_head_attention_key_bias_accumulator:>_
Iassignvariableop_68_adagrad_multi_head_attention_value_kernel_accumulator:>>Y
Gassignvariableop_69_adagrad_multi_head_attention_value_bias_accumulator:>j
Tassignvariableop_70_adagrad_multi_head_attention_attention_output_kernel_accumulator:>>`
Rassignvariableop_71_adagrad_multi_head_attention_attention_output_bias_accumulator:>
identity_73??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?&
value?&B?&IB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB`layer_with_weights-8/embeddings/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/4/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/5/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/6/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/7/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/8/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBLvariables/9/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/10/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBMvariables/11/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp+assignvariableop_movie_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_genres_vector_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp=assignvariableop_2_process_movie_embedding_with_genres_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp;assignvariableop_3_process_movie_embedding_with_genres_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_layer_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_layer_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_sex_embedding_embeddingsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp1assignvariableop_9_age_group_embedding_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_occupation_embedding_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_layer_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_layer_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_batch_normalization_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_batch_normalization_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp3assignvariableop_17_batch_normalization_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_batch_normalization_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp assignvariableop_20_dense_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_1_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_1_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_1_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_1_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_dense_3_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp assignvariableop_26_dense_3_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp5assignvariableop_27_multi_head_attention_query_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_multi_head_attention_query_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_key_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp1assignvariableop_30_multi_head_attention_key_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_multi_head_attention_value_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp3assignvariableop_32_multi_head_attention_value_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp@assignvariableop_33_multi_head_attention_attention_output_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp>assignvariableop_34_multi_head_attention_attention_output_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp assignvariableop_35_adagrad_iterIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp!assignvariableop_36_adagrad_decayIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adagrad_learning_rateIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpBassignvariableop_42_adagrad_movie_embedding_embeddings_accumulatorIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpRassignvariableop_43_adagrad_process_movie_embedding_with_genres_kernel_accumulatorIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpPassignvariableop_44_adagrad_process_movie_embedding_with_genres_bias_accumulatorIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpAassignvariableop_45_adagrad_layer_normalization_gamma_accumulatorIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adagrad_layer_normalization_beta_accumulatorIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adagrad_dense_kernel_accumulatorIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adagrad_dense_bias_accumulatorIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp@assignvariableop_49_adagrad_sex_embedding_embeddings_accumulatorIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpFassignvariableop_50_adagrad_age_group_embedding_embeddings_accumulatorIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpGassignvariableop_51_adagrad_occupation_embedding_embeddings_accumulatorIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpCassignvariableop_52_adagrad_layer_normalization_1_gamma_accumulatorIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpBassignvariableop_53_adagrad_layer_normalization_1_beta_accumulatorIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adagrad_dense_1_kernel_accumulatorIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adagrad_dense_1_bias_accumulatorIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpAassignvariableop_56_adagrad_batch_normalization_gamma_accumulatorIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp@assignvariableop_57_adagrad_batch_normalization_beta_accumulatorIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adagrad_dense_2_kernel_accumulatorIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adagrad_dense_2_bias_accumulatorIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpCassignvariableop_60_adagrad_batch_normalization_1_gamma_accumulatorIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpBassignvariableop_61_adagrad_batch_normalization_1_beta_accumulatorIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adagrad_dense_3_kernel_accumulatorIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adagrad_dense_3_bias_accumulatorIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpIassignvariableop_64_adagrad_multi_head_attention_query_kernel_accumulatorIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpGassignvariableop_65_adagrad_multi_head_attention_query_bias_accumulatorIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpGassignvariableop_66_adagrad_multi_head_attention_key_kernel_accumulatorIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpEassignvariableop_67_adagrad_multi_head_attention_key_bias_accumulatorIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpIassignvariableop_68_adagrad_multi_head_attention_value_kernel_accumulatorIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpGassignvariableop_69_adagrad_multi_head_attention_value_bias_accumulatorIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpTassignvariableop_70_adagrad_multi_head_attention_attention_output_kernel_accumulatorIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpRassignvariableop_71_adagrad_multi_head_attention_attention_output_bias_accumulatorIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_72Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_73IdentityIdentity_72:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_73Identity_73:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?$
?
-__inference_concatenate_3_layer_call_fn_25139
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26
	inputs_27
	inputs_28
	inputs_29
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/5:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/6:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/7:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/8:U	Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/9:V
R
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/10:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/11:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/12:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/13:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/14:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/15:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/16:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/17:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/18:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/19:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/20:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/21:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/22:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/23:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/24:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/25:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/26:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/27:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/28:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/29
??
?+
@__inference_model_layer_call_and_return_conditional_losses_24923
inputs_age_group
inputs_occupation
inputs_sequence_movie_ids
inputs_sequence_ratings

inputs_sex
inputs_target_movie_id
inputs_user_idA
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	9
&movie_embedding_embedding_lookup_24415:	?>7
$genres_vector_embedding_lookup_24420:	?W
Eprocess_movie_embedding_with_genres_tensordot_readvariableop_resource:P>Q
Cprocess_movie_embedding_with_genres_biasadd_readvariableop_resource:> 
tf___operators___add_addv2_yV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:>>H
6multi_head_attention_query_add_readvariableop_resource:>T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:>>F
4multi_head_attention_key_add_readvariableop_resource:>V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:>>H
6multi_head_attention_value_add_readvariableop_resource:>a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:>>O
Amulti_head_attention_attention_output_add_readvariableop_resource:>?
1layer_normalization_mul_3_readvariableop_resource:>=
/layer_normalization_add_readvariableop_resource:>9
'dense_tensordot_readvariableop_resource:>>3
%dense_biasadd_readvariableop_resource:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	6
$sex_embedding_embedding_lookup_24753:<
*age_group_embedding_embedding_lookup_24758:=
+occupation_embedding_embedding_lookup_24763:A
3layer_normalization_1_mul_3_readvariableop_resource:>?
1layer_normalization_1_add_readvariableop_resource:>:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?J
;batch_normalization_assignmovingavg_readvariableop_resource:	?L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?D
5batch_normalization_batchnorm_readvariableop_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??$age_group_embedding/embedding_lookup?#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?genres_vector/embedding_lookup? genres_vector/embedding_lookup_1?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_3/ReadVariableOp?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_3/ReadVariableOp? movie_embedding/embedding_lookup?"movie_embedding/embedding_lookup_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?%occupation_embedding/embedding_lookup?:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp?<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp?<process_movie_embedding_with_genres/Tensordot/ReadVariableOp?>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp?sex_embedding/embedding_lookup?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_sequence_movie_ids>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdinputs_sequence_movie_ids movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
 movie_embedding/embedding_lookupResourceGather&movie_embedding_embedding_lookup_24415$movie_index_lookup/Identity:output:0*
Tindices0	*9
_class/
-+loc:@movie_embedding/embedding_lookup/24415*+
_output_shapes
:?????????>*
dtype0?
)movie_embedding/embedding_lookup/IdentityIdentity)movie_embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@movie_embedding/embedding_lookup/24415*+
_output_shapes
:?????????>?
+movie_embedding/embedding_lookup/Identity_1Identity2movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>?
genres_vector/embedding_lookupResourceGather$genres_vector_embedding_lookup_24420$movie_index_lookup/Identity:output:0*
Tindices0	*7
_class-
+)loc:@genres_vector/embedding_lookup/24420*+
_output_shapes
:?????????*
dtype0?
'genres_vector/embedding_lookup/IdentityIdentity'genres_vector/embedding_lookup:output:0*
T0*7
_class-
+)loc:@genres_vector/embedding_lookup/24420*+
_output_shapes
:??????????
)genres_vector/embedding_lookup/Identity_1Identity0genres_vector/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV24movie_embedding/embedding_lookup/Identity_1:output:02genres_vector/embedding_lookup/Identity_1:output:0"concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
<process_movie_embedding_with_genres/Tensordot/ReadVariableOpReadVariableOpEprocess_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0|
2process_movie_embedding_with_genres/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
2process_movie_embedding_with_genres/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
3process_movie_embedding_with_genres/Tensordot/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:}
;process_movie_embedding_with_genres/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot/GatherV2GatherV2<process_movie_embedding_with_genres/Tensordot/Shape:output:0;process_movie_embedding_with_genres/Tensordot/free:output:0Dprocess_movie_embedding_with_genres/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=process_movie_embedding_with_genres/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot/GatherV2_1GatherV2<process_movie_embedding_with_genres/Tensordot/Shape:output:0;process_movie_embedding_with_genres/Tensordot/axes:output:0Fprocess_movie_embedding_with_genres/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3process_movie_embedding_with_genres/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2process_movie_embedding_with_genres/Tensordot/ProdProd?process_movie_embedding_with_genres/Tensordot/GatherV2:output:0<process_movie_embedding_with_genres/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5process_movie_embedding_with_genres/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
4process_movie_embedding_with_genres/Tensordot/Prod_1ProdAprocess_movie_embedding_with_genres/Tensordot/GatherV2_1:output:0>process_movie_embedding_with_genres/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9process_movie_embedding_with_genres/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4process_movie_embedding_with_genres/Tensordot/concatConcatV2;process_movie_embedding_with_genres/Tensordot/free:output:0;process_movie_embedding_with_genres/Tensordot/axes:output:0Bprocess_movie_embedding_with_genres/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3process_movie_embedding_with_genres/Tensordot/stackPack;process_movie_embedding_with_genres/Tensordot/Prod:output:0=process_movie_embedding_with_genres/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
7process_movie_embedding_with_genres/Tensordot/transpose	Transposeconcatenate_2/concat:output:0=process_movie_embedding_with_genres/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
5process_movie_embedding_with_genres/Tensordot/ReshapeReshape;process_movie_embedding_with_genres/Tensordot/transpose:y:0<process_movie_embedding_with_genres/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
4process_movie_embedding_with_genres/Tensordot/MatMulMatMul>process_movie_embedding_with_genres/Tensordot/Reshape:output:0Dprocess_movie_embedding_with_genres/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>
5process_movie_embedding_with_genres/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>}
;process_movie_embedding_with_genres/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot/concat_1ConcatV2?process_movie_embedding_with_genres/Tensordot/GatherV2:output:0>process_movie_embedding_with_genres/Tensordot/Const_2:output:0Dprocess_movie_embedding_with_genres/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
-process_movie_embedding_with_genres/TensordotReshape>process_movie_embedding_with_genres/Tensordot/MatMul:product:0?process_movie_embedding_with_genres/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
:process_movie_embedding_with_genres/BiasAdd/ReadVariableOpReadVariableOpCprocess_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
+process_movie_embedding_with_genres/BiasAddBiasAdd6process_movie_embedding_with_genres/Tensordot:output:0Bprocess_movie_embedding_with_genres/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
(process_movie_embedding_with_genres/ReluRelu4process_movie_embedding_with_genres/BiasAdd:output:0*
T0*+
_output_shapes
:?????????>?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_target_movie_id>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdinputs_target_movie_id"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV26process_movie_embedding_with_genres/Relu:activations:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimsinputs_sequence_ratings&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
"movie_embedding/embedding_lookup_1ResourceGather&movie_embedding_embedding_lookup_24415&movie_index_lookup/Identity_1:output:0*
Tindices0	*9
_class/
-+loc:@movie_embedding/embedding_lookup/24415*+
_output_shapes
:?????????>*
dtype0?
+movie_embedding/embedding_lookup_1/IdentityIdentity+movie_embedding/embedding_lookup_1:output:0*
T0*9
_class/
-+loc:@movie_embedding/embedding_lookup/24415*+
_output_shapes
:?????????>?
-movie_embedding/embedding_lookup_1/Identity_1Identity4movie_embedding/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????>?
 genres_vector/embedding_lookup_1ResourceGather$genres_vector_embedding_lookup_24420&movie_index_lookup/Identity_1:output:0*
Tindices0	*7
_class-
+)loc:@genres_vector/embedding_lookup/24420*+
_output_shapes
:?????????*
dtype0?
)genres_vector/embedding_lookup_1/IdentityIdentity)genres_vector/embedding_lookup_1:output:0*
T0*7
_class-
+)loc:@genres_vector/embedding_lookup/24420*+
_output_shapes
:??????????
+genres_vector/embedding_lookup_1/Identity_1Identity2genres_vector/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:??????????
multiply/mulMultf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*+
_output_shapes
:?????????>[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV26movie_embedding/embedding_lookup_1/Identity_1:output:04genres_vector/embedding_lookup_1/Identity_1:output:0"concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
tf.unstack/unstackUnpackmultiply/mul:z:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOpReadVariableOpEprocess_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0~
4process_movie_embedding_with_genres/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4process_movie_embedding_with_genres/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5process_movie_embedding_with_genres/Tensordot_1/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:
=process_movie_embedding_with_genres/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot_1/GatherV2GatherV2>process_movie_embedding_with_genres/Tensordot_1/Shape:output:0=process_movie_embedding_with_genres/Tensordot_1/free:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?process_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:process_movie_embedding_with_genres/Tensordot_1/GatherV2_1GatherV2>process_movie_embedding_with_genres/Tensordot_1/Shape:output:0=process_movie_embedding_with_genres/Tensordot_1/axes:output:0Hprocess_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5process_movie_embedding_with_genres/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4process_movie_embedding_with_genres/Tensordot_1/ProdProdAprocess_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0>process_movie_embedding_with_genres/Tensordot_1/Const:output:0*
T0*
_output_shapes
: ?
7process_movie_embedding_with_genres/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6process_movie_embedding_with_genres/Tensordot_1/Prod_1ProdCprocess_movie_embedding_with_genres/Tensordot_1/GatherV2_1:output:0@process_movie_embedding_with_genres/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: }
;process_movie_embedding_with_genres/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot_1/concatConcatV2=process_movie_embedding_with_genres/Tensordot_1/free:output:0=process_movie_embedding_with_genres/Tensordot_1/axes:output:0Dprocess_movie_embedding_with_genres/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5process_movie_embedding_with_genres/Tensordot_1/stackPack=process_movie_embedding_with_genres/Tensordot_1/Prod:output:0?process_movie_embedding_with_genres/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9process_movie_embedding_with_genres/Tensordot_1/transpose	Transposeconcatenate_1/concat:output:0?process_movie_embedding_with_genres/Tensordot_1/concat:output:0*
T0*+
_output_shapes
:?????????P?
7process_movie_embedding_with_genres/Tensordot_1/ReshapeReshape=process_movie_embedding_with_genres/Tensordot_1/transpose:y:0>process_movie_embedding_with_genres/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:???????????????????
6process_movie_embedding_with_genres/Tensordot_1/MatMulMatMul@process_movie_embedding_with_genres/Tensordot_1/Reshape:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
7process_movie_embedding_with_genres/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>
=process_movie_embedding_with_genres/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot_1/concat_1ConcatV2Aprocess_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0@process_movie_embedding_with_genres/Tensordot_1/Const_2:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/process_movie_embedding_with_genres/Tensordot_1Reshape@process_movie_embedding_with_genres/Tensordot_1/MatMul:product:0Aprocess_movie_embedding_with_genres/Tensordot_1/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOpReadVariableOpCprocess_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
-process_movie_embedding_with_genres/BiasAdd_1BiasAdd8process_movie_embedding_with_genres/Tensordot_1:output:0Dprocess_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
*process_movie_embedding_with_genres/Relu_1Relu6process_movie_embedding_with_genres/BiasAdd_1:output:0*
T0*+
_output_shapes
:?????????>[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?

concatenate_3/concatConcatV2$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:08process_movie_embedding_with_genres/Relu_1:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????>?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumconcatenate_3/concat:output:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumconcatenate_3/concat:output:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumconcatenate_3/concat:output:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????o
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:??????????
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0x
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????>r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>?
add/addAddV2concatenate_3/concat:output:0dropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>T
layer_normalization/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization/ReshapeReshapeadd/add:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:?????????>?
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>|
leaky_re_lu/LeakyRelu	LeakyRelulayer_normalization/add:z:0*+
_output_shapes
:?????????>*
alpha%???>?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:>>*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
dense/Tensordot/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose#leaky_re_lu/LeakyRelu:activations:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????>?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_occupation;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNdinputs_occupationstring_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0#^movie_index_lookup/Assert_1/Assert*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_age_group;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNdinputs_age_groupstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handle
inputs_sex9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNd
inputs_sexstring_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:?????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????>]
dropout_1/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>?
sex_embedding/embedding_lookupResourceGather$sex_embedding_embedding_lookup_24753string_lookup/Identity:output:0*
Tindices0	*7
_class-
+)loc:@sex_embedding/embedding_lookup/24753*+
_output_shapes
:?????????*
dtype0?
'sex_embedding/embedding_lookup/IdentityIdentity'sex_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@sex_embedding/embedding_lookup/24753*+
_output_shapes
:??????????
)sex_embedding/embedding_lookup/Identity_1Identity0sex_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
$age_group_embedding/embedding_lookupResourceGather*age_group_embedding_embedding_lookup_24758!string_lookup_1/Identity:output:0*
Tindices0	*=
_class3
1/loc:@age_group_embedding/embedding_lookup/24758*+
_output_shapes
:?????????*
dtype0?
-age_group_embedding/embedding_lookup/IdentityIdentity-age_group_embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@age_group_embedding/embedding_lookup/24758*+
_output_shapes
:??????????
/age_group_embedding/embedding_lookup/Identity_1Identity6age_group_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
%occupation_embedding/embedding_lookupResourceGather+occupation_embedding_embedding_lookup_24763!string_lookup_2/Identity:output:0*
Tindices0	*>
_class4
20loc:@occupation_embedding/embedding_lookup/24763*+
_output_shapes
:?????????*
dtype0?
.occupation_embedding/embedding_lookup/IdentityIdentity.occupation_embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@occupation_embedding/embedding_lookup/24763*+
_output_shapes
:??????????
0occupation_embedding/embedding_lookup/Identity_1Identity7occupation_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
	add_1/addAddV2layer_normalization/add:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV22sex_embedding/embedding_lookup/Identity_1:output:08age_group_embedding/embedding_lookup/Identity_1:output:09occupation_embedding/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????X
layer_normalization_1/ShapeShapeadd_1/add:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_1/ReshapeReshapeadd_1/add:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:?????????y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:?????????>?
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  ?
flatten/ReshapeReshapelayer_normalization_1/add:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????X
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_4/concatConcatV2flatten/Reshape:output:0reshape/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMulconcatenate_4/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
 batch_normalization/moments/meanMeandense_1/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	??
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Muldense_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
leaky_re_lu_1/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_2/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:??????????l
dropout_2/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
"batch_normalization_1/moments/meanMeandense_2/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	??
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_2/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:???????????
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 ?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:??
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:??
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
leaky_re_lu_2/LeakyRelu	LeakyRelu)batch_normalization_1/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_3/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:??????????l
dropout_3/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:?
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^age_group_embedding/embedding_lookup$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^genres_vector/embedding_lookup!^genres_vector/embedding_lookup_1'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp!^movie_embedding/embedding_lookup#^movie_embedding/embedding_lookup_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV29^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp&^occupation_embedding/embedding_lookup;^process_movie_embedding_with_genres/BiasAdd/ReadVariableOp=^process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp=^process_movie_embedding_with_genres/Tensordot/ReadVariableOp?^process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp^sex_embedding/embedding_lookup^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$age_group_embedding/embedding_lookup$age_group_embedding/embedding_lookup2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
genres_vector/embedding_lookupgenres_vector/embedding_lookup2D
 genres_vector/embedding_lookup_1 genres_vector/embedding_lookup_12P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2D
 movie_embedding/embedding_lookup movie_embedding/embedding_lookup2H
"movie_embedding/embedding_lookup_1"movie_embedding/embedding_lookup_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2N
%occupation_embedding/embedding_lookup%occupation_embedding/embedding_lookup2x
:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp2|
<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp2|
<process_movie_embedding_with_genres/Tensordot/ReadVariableOp<process_movie_embedding_with_genres/Tensordot/ReadVariableOp2?
>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp2@
sex_embedding/embedding_lookupsex_embedding/embedding_lookup2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/age_group:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/sequence_movie_ids:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/sequence_ratings:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/target_movie_id:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_25952
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name12*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_25510

inputs	(
embedding_lookup_25504:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_25504inputs*
Tindices0	*)
_class
loc:@embedding_lookup/25504*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/25504*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_22001

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
:
__inference__creator_25988
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name98*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_25000
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :{
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
@__inference_model_layer_call_and_return_conditional_losses_23359
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_idA
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	(
movie_embedding_23112:	?>&
genres_vector_23115:	?;
)process_movie_embedding_with_genres_23119:P>7
)process_movie_embedding_with_genres_23121:> 
tf___operators___add_addv2_y0
multi_head_attention_23235:>>,
multi_head_attention_23237:>0
multi_head_attention_23239:>>,
multi_head_attention_23241:>0
multi_head_attention_23243:>>,
multi_head_attention_23245:>0
multi_head_attention_23247:>>(
multi_head_attention_23249:>'
layer_normalization_23254:>'
layer_normalization_23256:>
dense_23260:>>
dense_23262:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	%
sex_embedding_23302:+
age_group_embedding_23305:,
occupation_embedding_23308:)
layer_normalization_1_23313:>)
layer_normalization_1_23315:>!
dense_1_23321:
??
dense_1_23323:	?(
batch_normalization_23326:	?(
batch_normalization_23328:	?(
batch_normalization_23330:	?(
batch_normalization_23332:	?!
dense_2_23337:
??
dense_2_23339:	?*
batch_normalization_1_23342:	?*
batch_normalization_1_23344:	?*
batch_normalization_1_23346:	?*
batch_normalization_1_23348:	? 
dense_3_23353:	?
dense_3_23355:
identity??+age_group_embedding/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?%genres_vector/StatefulPartitionedCall?'genres_vector/StatefulPartitionedCall_1?+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?)movie_embedding/StatefulPartitionedCall_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?,multi_head_attention/StatefulPartitionedCall?,occupation_embedding/StatefulPartitionedCall?;process_movie_embedding_with_genres/StatefulPartitionedCall?=process_movie_embedding_with_genres/StatefulPartitionedCall_1?%sex_embedding/StatefulPartitionedCall?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handlesequence_movie_ids>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdsequence_movie_ids movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0movie_embedding_23112*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291?
%genres_vector/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0genres_vector_23115*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304?
concatenate_2/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0.genres_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315?
;process_movie_embedding_with_genres/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0)process_movie_embedding_with_genres_23119)process_movie_embedding_with_genres_23121*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handletarget_movie_id>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdtarget_movie_id"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2Dprocess_movie_embedding_with_genres/StatefulPartitionedCall:output:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimssequence_ratings&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
)movie_embedding/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0movie_embedding_23112*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376?
'genres_vector/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0genres_vector_23115*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387?
multiply/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_multiply_layer_call_and_return_conditional_losses_21396?
concatenate_1/PartitionedCallPartitionedCall2movie_embedding/StatefulPartitionedCall_1:output:00genres_vector/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405?
tf.unstack/unstackUnpack!multiply/PartitionedCall:output:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
=process_movie_embedding_with_genres/StatefulPartitionedCall_1StatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)process_movie_embedding_with_genres_23119)process_movie_embedding_with_genres_23121*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524?
concatenate_3/PartitionedCallPartitionedCall$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:0Fprocess_movie_embedding_with_genres/StatefulPartitionedCall_1:output:0*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563?
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0&concatenate_3/PartitionedCall:output:0multi_head_attention_23235multi_head_attention_23237multi_head_attention_23239multi_head_attention_23241multi_head_attention_23243multi_head_attention_23245multi_head_attention_23247multi_head_attention_23249*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_21600?
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_21623?
add/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0 dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_21631?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_23254layer_normalization_23256*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680?
leaky_re_lu/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691?
dense/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0dense_23260dense_23262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21723?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handle
occupation;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNd
occupationstring_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0#^movie_index_lookup/Assert_1/Assert*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handle	age_group;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNd	age_groupstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlesex9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNdsexstring_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_21770?
%sex_embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0sex_embedding_23302*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781?
+age_group_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0age_group_embedding_23305*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794?
,occupation_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0occupation_embedding_23308*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807?
add_1/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_21817?
concatenate/PartitionedCallPartitionedCall.sex_embedding/StatefulPartitionedCall:output:04age_group_embedding/StatefulPartitionedCall:output:05occupation_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21827?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_1_23313layer_normalization_1_23315*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876?
flatten/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_21888?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_21902?
concatenate_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_1_23321dense_1_23323*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_21923?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_23326batch_normalization_23328batch_normalization_23330batch_normalization_23332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21111?
leaky_re_lu_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_21950?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_23337dense_2_23339*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_21962?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_1_23342batch_normalization_1_23344batch_normalization_1_23346batch_normalization_1_23348*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21193?
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982?
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_21989?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_23353dense_3_23355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_22001w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp,^age_group_embedding/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall&^genres_vector/StatefulPartitionedCall(^genres_vector/StatefulPartitionedCall_1,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall*^movie_embedding/StatefulPartitionedCall_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV2-^multi_head_attention/StatefulPartitionedCall-^occupation_embedding/StatefulPartitionedCall<^process_movie_embedding_with_genres/StatefulPartitionedCall>^process_movie_embedding_with_genres/StatefulPartitionedCall_1&^sex_embedding/StatefulPartitionedCall^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+age_group_embedding/StatefulPartitionedCall+age_group_embedding/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2N
%genres_vector/StatefulPartitionedCall%genres_vector/StatefulPartitionedCall2R
'genres_vector/StatefulPartitionedCall_1'genres_vector/StatefulPartitionedCall_12Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2V
)movie_embedding/StatefulPartitionedCall_1)movie_embedding/StatefulPartitionedCall_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2\
,occupation_embedding/StatefulPartitionedCall,occupation_embedding/StatefulPartitionedCall2z
;process_movie_embedding_with_genres/StatefulPartitionedCall;process_movie_embedding_with_genres/StatefulPartitionedCall2~
=process_movie_embedding_with_genres/StatefulPartitionedCall_1=process_movie_embedding_with_genres/StatefulPartitionedCall_12N
%sex_embedding/StatefulPartitionedCall%sex_embedding/StatefulPartitionedCall2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_movie_embedding_layer_call_fn_24937

inputs	
unknown:	?>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_25310

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?1
?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_22402	
query	
valueA
+query_einsum_einsum_readvariableop_resource:>>3
!query_add_readvariableop_resource:>?
)key_einsum_einsum_readvariableop_resource:>>1
key_add_readvariableop_resource:>A
+value_einsum_einsum_readvariableop_resource:>>3
!value_add_readvariableop_resource:>L
6attention_output_einsum_einsum_readvariableop_resource:>>:
,attention_output_add_readvariableop_resource:>
identity??#attention_output/add/ReadVariableOp?-attention_output/einsum/Einsum/ReadVariableOp?key/add/ReadVariableOp? key/einsum/Einsum/ReadVariableOp?query/add/ReadVariableOp?"query/einsum/Einsum/ReadVariableOp?value/add/ReadVariableOp?"value/einsum/Einsum/ReadVariableOp?
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:?????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:?????????>?
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?#
?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_25174
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26
	inputs_27
	inputs_28
	inputs_29
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????>[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/5:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/6:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/7:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/8:U	Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/9:V
R
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/10:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/11:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/12:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/13:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/14:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/15:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/16:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/17:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/18:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/19:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/20:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/21:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/22:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/23:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/24:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/25:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/26:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/27:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/28:VR
+
_output_shapes
:?????????>
#
_user_specified_name	inputs/29
?
?
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387

inputs	)
embedding_lookup_21381:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21381inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21381*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21381*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25018

inputs
unknown:P>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
__inference__initializer_259785
1key_value_init54_lookuptableimportv2_table_handle-
)key_value_init54_lookuptableimportv2_keys/
+key_value_init54_lookuptableimportv2_values	
identity??$key_value_init54/LookupTableImportV2?
$key_value_init54/LookupTableImportV2LookupTableImportV21key_value_init54_lookuptableimportv2_table_handle)key_value_init54_lookuptableimportv2_keys+key_value_init54_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init54/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init54/LookupTableImportV2$key_value_init54/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
3__inference_age_group_embedding_layer_call_fn_25501

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_260175
1key_value_init11_lookuptableimportv2_table_handle-
)key_value_init11_lookuptableimportv2_keys/
+key_value_init11_lookuptableimportv2_values	
identity??$key_value_init11/LookupTableImportV2?
$key_value_init11/LookupTableImportV2LookupTableImportV21key_value_init11_lookuptableimportv2_table_handle)key_value_init11_lookuptableimportv2_keys+key_value_init11_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init11/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init11/LookupTableImportV2$key_value_init11/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_21989

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???>`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
^
B__inference_reshape_layer_call_and_return_conditional_losses_25625

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_22008

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6A
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	(
movie_embedding_21292:	?>&
genres_vector_21305:	?;
)process_movie_embedding_with_genres_21349:P>7
)process_movie_embedding_with_genres_21351:> 
tf___operators___add_addv2_y0
multi_head_attention_21601:>>,
multi_head_attention_21603:>0
multi_head_attention_21605:>>,
multi_head_attention_21607:>0
multi_head_attention_21609:>>,
multi_head_attention_21611:>0
multi_head_attention_21613:>>(
multi_head_attention_21615:>'
layer_normalization_21681:>'
layer_normalization_21683:>
dense_21724:>>
dense_21726:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	%
sex_embedding_21782:+
age_group_embedding_21795:,
occupation_embedding_21808:)
layer_normalization_1_21877:>)
layer_normalization_1_21879:>!
dense_1_21924:
??
dense_1_21926:	?(
batch_normalization_21929:	?(
batch_normalization_21931:	?(
batch_normalization_21933:	?(
batch_normalization_21935:	?!
dense_2_21963:
??
dense_2_21965:	?*
batch_normalization_1_21968:	?*
batch_normalization_1_21970:	?*
batch_normalization_1_21972:	?*
batch_normalization_1_21974:	? 
dense_3_22002:	?
dense_3_22004:
identity??+age_group_embedding/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?%genres_vector/StatefulPartitionedCall?'genres_vector/StatefulPartitionedCall_1?+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?)movie_embedding/StatefulPartitionedCall_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?,multi_head_attention/StatefulPartitionedCall?,occupation_embedding/StatefulPartitionedCall?;process_movie_embedding_with_genres/StatefulPartitionedCall?=process_movie_embedding_with_genres/StatefulPartitionedCall_1?%sex_embedding/StatefulPartitionedCall?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdinputs_2 movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0movie_embedding_21292*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291?
%genres_vector/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0genres_vector_21305*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304?
concatenate_2/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0.genres_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315?
;process_movie_embedding_with_genres/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0)process_movie_embedding_with_genres_21349)process_movie_embedding_with_genres_21351*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_5>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdinputs_5"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2Dprocess_movie_embedding_with_genres/StatefulPartitionedCall:output:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimsinputs_3&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
)movie_embedding/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0movie_embedding_21292*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376?
'genres_vector/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0genres_vector_21305*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387?
multiply/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_multiply_layer_call_and_return_conditional_losses_21396?
concatenate_1/PartitionedCallPartitionedCall2movie_embedding/StatefulPartitionedCall_1:output:00genres_vector/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405?
tf.unstack/unstackUnpack!multiply/PartitionedCall:output:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
=process_movie_embedding_with_genres/StatefulPartitionedCall_1StatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)process_movie_embedding_with_genres_21349)process_movie_embedding_with_genres_21351*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524?
concatenate_3/PartitionedCallPartitionedCall$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:0Fprocess_movie_embedding_with_genres/StatefulPartitionedCall_1:output:0*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563?
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0&concatenate_3/PartitionedCall:output:0multi_head_attention_21601multi_head_attention_21603multi_head_attention_21605multi_head_attention_21607multi_head_attention_21609multi_head_attention_21611multi_head_attention_21613multi_head_attention_21615*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_21600?
dropout/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_21623?
add/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0 dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_21631?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_21681layer_normalization_21683*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680?
leaky_re_lu/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691?
dense/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0dense_21724dense_21726*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21723?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_1;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNdinputs_1string_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0#^movie_index_lookup/Assert_1/Assert*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNdinputsstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_49string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNdinputs_4string_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_21770?
%sex_embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0sex_embedding_21782*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781?
+age_group_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0age_group_embedding_21795*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794?
,occupation_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0occupation_embedding_21808*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807?
add_1/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_21817?
concatenate/PartitionedCallPartitionedCall.sex_embedding/StatefulPartitionedCall:output:04age_group_embedding/StatefulPartitionedCall:output:05occupation_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21827?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_1_21877layer_normalization_1_21879*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876?
flatten/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_21888?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_21902?
concatenate_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_1_21924dense_1_21926*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_21923?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_21929batch_normalization_21931batch_normalization_21933batch_normalization_21935*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21111?
leaky_re_lu_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_21950?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_21963dense_2_21965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_21962?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_1_21968batch_normalization_1_21970batch_normalization_1_21972batch_normalization_1_21974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21193?
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982?
dropout_3/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_21989?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_22002dense_3_22004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_22001w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp,^age_group_embedding/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall&^genres_vector/StatefulPartitionedCall(^genres_vector/StatefulPartitionedCall_1,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall*^movie_embedding/StatefulPartitionedCall_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV2-^multi_head_attention/StatefulPartitionedCall-^occupation_embedding/StatefulPartitionedCall<^process_movie_embedding_with_genres/StatefulPartitionedCall>^process_movie_embedding_with_genres/StatefulPartitionedCall_1&^sex_embedding/StatefulPartitionedCall^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+age_group_embedding/StatefulPartitionedCall+age_group_embedding/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2N
%genres_vector/StatefulPartitionedCall%genres_vector/StatefulPartitionedCall2R
'genres_vector/StatefulPartitionedCall_1'genres_vector/StatefulPartitionedCall_12Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2V
)movie_embedding/StatefulPartitionedCall_1)movie_embedding/StatefulPartitionedCall_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2\
,occupation_embedding/StatefulPartitionedCall,occupation_embedding/StatefulPartitionedCall2z
;process_movie_embedding_with_genres/StatefulPartitionedCall;process_movie_embedding_with_genres/StatefulPartitionedCall2~
=process_movie_embedding_with_genres/StatefulPartitionedCall_1=process_movie_embedding_with_genres/StatefulPartitionedCall_12N
%sex_embedding/StatefulPartitionedCall%sex_embedding/StatefulPartitionedCall2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_22331

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?

#__inference_signature_wrapper_23730
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_id
unknown
	unknown_0	
	unknown_1:	?>
	unknown_2:	?
	unknown_3:P>
	unknown_4:>
	unknown_5
	unknown_6:>>
	unknown_7:>
	unknown_8:>>
	unknown_9:> 

unknown_10:>>

unknown_11:> 

unknown_12:>>

unknown_13:>

unknown_14:>

unknown_15:>

unknown_16:>>

unknown_17:>

unknown_18

unknown_19	

unknown_20

unknown_21	

unknown_22

unknown_23	

unknown_24:

unknown_25:

unknown_26:

unknown_27:>

unknown_28:>

unknown_29:
??

unknown_30:	?

unknown_31:	?

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*E
_read_only_resource_inputs'
%#	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_21087o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_25934
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name147*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?$
?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_25390

inputs+
mul_3_readvariableop_resource:>)
add_readvariableop_resource:>
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????>L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????>n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:>*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:>*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????>r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
e
+__inference_concatenate_layer_call_fn_25589
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21827d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2
??
?
@__inference_model_layer_call_and_return_conditional_losses_23627
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_idA
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	(
movie_embedding_23380:	?>&
genres_vector_23383:	?;
)process_movie_embedding_with_genres_23387:P>7
)process_movie_embedding_with_genres_23389:> 
tf___operators___add_addv2_y0
multi_head_attention_23503:>>,
multi_head_attention_23505:>0
multi_head_attention_23507:>>,
multi_head_attention_23509:>0
multi_head_attention_23511:>>,
multi_head_attention_23513:>0
multi_head_attention_23515:>>(
multi_head_attention_23517:>'
layer_normalization_23522:>'
layer_normalization_23524:>
dense_23528:>>
dense_23530:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	%
sex_embedding_23570:+
age_group_embedding_23573:,
occupation_embedding_23576:)
layer_normalization_1_23581:>)
layer_normalization_1_23583:>!
dense_1_23589:
??
dense_1_23591:	?(
batch_normalization_23594:	?(
batch_normalization_23596:	?(
batch_normalization_23598:	?(
batch_normalization_23600:	?!
dense_2_23605:
??
dense_2_23607:	?*
batch_normalization_1_23610:	?*
batch_normalization_1_23612:	?*
batch_normalization_1_23614:	?*
batch_normalization_1_23616:	? 
dense_3_23621:	?
dense_3_23623:
identity??+age_group_embedding/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?%genres_vector/StatefulPartitionedCall?'genres_vector/StatefulPartitionedCall_1?+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?)movie_embedding/StatefulPartitionedCall_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?,multi_head_attention/StatefulPartitionedCall?,occupation_embedding/StatefulPartitionedCall?;process_movie_embedding_with_genres/StatefulPartitionedCall?=process_movie_embedding_with_genres/StatefulPartitionedCall_1?%sex_embedding/StatefulPartitionedCall?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handlesequence_movie_ids>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdsequence_movie_ids movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0movie_embedding_23380*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291?
%genres_vector/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0genres_vector_23383*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304?
concatenate_2/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0.genres_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315?
;process_movie_embedding_with_genres/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0)process_movie_embedding_with_genres_23387)process_movie_embedding_with_genres_23389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handletarget_movie_id>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdtarget_movie_id"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2Dprocess_movie_embedding_with_genres/StatefulPartitionedCall:output:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimssequence_ratings&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
)movie_embedding/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0movie_embedding_23380*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376?
'genres_vector/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0genres_vector_23383*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387?
multiply/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_multiply_layer_call_and_return_conditional_losses_21396?
concatenate_1/PartitionedCallPartitionedCall2movie_embedding/StatefulPartitionedCall_1:output:00genres_vector/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405?
tf.unstack/unstackUnpack!multiply/PartitionedCall:output:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
=process_movie_embedding_with_genres/StatefulPartitionedCall_1StatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)process_movie_embedding_with_genres_23387)process_movie_embedding_with_genres_23389*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524?
concatenate_3/PartitionedCallPartitionedCall$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:0Fprocess_movie_embedding_with_genres/StatefulPartitionedCall_1:output:0*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563?
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0&concatenate_3/PartitionedCall:output:0multi_head_attention_23503multi_head_attention_23505multi_head_attention_23507multi_head_attention_23509multi_head_attention_23511multi_head_attention_23513multi_head_attention_23515multi_head_attention_23517*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_22402?
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0#^movie_index_lookup/Assert_1/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_22331?
add/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_21631?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_23522layer_normalization_23524*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680?
leaky_re_lu/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691?
dense/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0dense_23528dense_23530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21723?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handle
occupation;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNd
occupationstring_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0 ^dropout/StatefulPartitionedCall*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handle	age_group;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNd	age_groupstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlesex9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNdsexstring_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0^string_lookup/Assert/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_22275?
%sex_embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0sex_embedding_23570*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781?
+age_group_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0age_group_embedding_23573*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794?
,occupation_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0occupation_embedding_23576*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807?
add_1/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_21817?
concatenate/PartitionedCallPartitionedCall.sex_embedding/StatefulPartitionedCall:output:04age_group_embedding/StatefulPartitionedCall:output:05occupation_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21827?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_1_23581layer_normalization_1_23583*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876?
flatten/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_21888?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_21902?
concatenate_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_1_23589dense_1_23591*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_21923?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_23594batch_normalization_23596batch_normalization_23598batch_normalization_23600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21158?
leaky_re_lu_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_22168?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_23605dense_2_23607*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_21962?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_1_23610batch_normalization_1_23612batch_normalization_1_23614batch_normalization_1_23616*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21240?
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_22129?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_23621dense_3_23623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_22001w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^age_group_embedding/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall&^genres_vector/StatefulPartitionedCall(^genres_vector/StatefulPartitionedCall_1,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall*^movie_embedding/StatefulPartitionedCall_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV2-^multi_head_attention/StatefulPartitionedCall-^occupation_embedding/StatefulPartitionedCall<^process_movie_embedding_with_genres/StatefulPartitionedCall>^process_movie_embedding_with_genres/StatefulPartitionedCall_1&^sex_embedding/StatefulPartitionedCall^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+age_group_embedding/StatefulPartitionedCall+age_group_embedding/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2N
%genres_vector/StatefulPartitionedCall%genres_vector/StatefulPartitionedCall2R
'genres_vector/StatefulPartitionedCall_1'genres_vector/StatefulPartitionedCall_12Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2V
)movie_embedding/StatefulPartitionedCall_1)movie_embedding/StatefulPartitionedCall_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2\
,occupation_embedding/StatefulPartitionedCall,occupation_embedding/StatefulPartitionedCall2z
;process_movie_embedding_with_genres/StatefulPartitionedCall;process_movie_embedding_with_genres/StatefulPartitionedCall2~
=process_movie_embedding_with_genres/StatefulPartitionedCall_1=process_movie_embedding_with_genres/StatefulPartitionedCall_12N
%sex_embedding/StatefulPartitionedCall%sex_embedding/StatefulPartitionedCall2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
4__inference_multi_head_attention_layer_call_fn_25196	
query	
value
unknown:>>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_21600s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?
?
__inference__initializer_259426
2key_value_init146_lookuptableimportv2_table_handle.
*key_value_init146_lookuptableimportv2_keys0
,key_value_init146_lookuptableimportv2_values	
identity??%key_value_init146/LookupTableImportV2?
%key_value_init146/LookupTableImportV2LookupTableImportV22key_value_init146_lookuptableimportv2_table_handle*key_value_init146_lookuptableimportv2_keys,key_value_init146_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init146/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init146/LookupTableImportV2%key_value_init146/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
H__inference_sex_embedding_layer_call_and_return_conditional_losses_25494

inputs	(
embedding_lookup_25488:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_25488inputs*
Tindices0	*)
_class
loc:@embedding_lookup/25488*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/25488*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21111

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25737

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_25747

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???>`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_25752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_21950a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_3_layer_call_fn_25888

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_21989a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_25898

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference__initializer_259605
1key_value_init11_lookuptableimportv2_table_handle-
)key_value_init11_lookuptableimportv2_keys/
+key_value_init11_lookuptableimportv2_values	
identity??$key_value_init11/LookupTableImportV2?
$key_value_init11/LookupTableImportV2LookupTableImportV21key_value_init11_lookuptableimportv2_table_handle)key_value_init11_lookuptableimportv2_keys+key_value_init11_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init11/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init11/LookupTableImportV2$key_value_init11/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
t
H__inference_concatenate_4_layer_call_and_return_conditional_losses_25638
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????:?????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
G
+__inference_leaky_re_lu_layer_call_fn_25395

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :y
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_3_layer_call_fn_25919

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_22001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????:?????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_genres_vector_layer_call_fn_24962

inputs	
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
C__inference_multiply_layer_call_and_return_conditional_losses_21396

inputs
inputs_1
identityR
mulMulinputsinputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_25444

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_21770d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
:
__inference__creator_25970
identity??
hash_tablej

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name55*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
-__inference_sex_embedding_layer_call_fn_25485

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_reshape_layer_call_fn_25613

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_21902`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_25793

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_leaky_re_lu_1_layer_call_fn_25742

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_layer_normalization_layer_call_fn_25343

inputs
unknown:>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?	
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_25910

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_21623

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_25597
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:?????????:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348

inputs3
!tensordot_readvariableop_resource:P>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????>e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
Q
%__inference_add_1_layer_call_fn_25472
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_21817d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1
?
?
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304

inputs	)
embedding_lookup_21298:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21298inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21298*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21298*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21158

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_2_layer_call_fn_24993
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_25657

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25873

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_21888

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_25757

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_22168p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21193

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
o
C__inference_multiply_layer_call_and_return_conditional_losses_25092
inputs_0
inputs_1
identityT
mulMulinputs_0inputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524

inputs3
!tensordot_readvariableop_resource:P>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????>e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24946

inputs	)
embedding_lookup_24940:	?>
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_24940inputs*
Tindices0	*)
_class
loc:@embedding_lookup/24940*+
_output_shapes
:?????????>*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/24940*+
_output_shapes
:?????????>?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????>Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_21770

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????>_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????>"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943

inputs
identityX
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:??????????*
alpha%???>`
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_22168

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?*
@__inference_model_layer_call_and_return_conditional_losses_24394
inputs_age_group
inputs_occupation
inputs_sequence_movie_ids
inputs_sequence_ratings

inputs_sex
inputs_target_movie_id
inputs_user_idA
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	9
&movie_embedding_embedding_lookup_23949:	?>7
$genres_vector_embedding_lookup_23954:	?W
Eprocess_movie_embedding_with_genres_tensordot_readvariableop_resource:P>Q
Cprocess_movie_embedding_with_genres_biasadd_readvariableop_resource:> 
tf___operators___add_addv2_yV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:>>H
6multi_head_attention_query_add_readvariableop_resource:>T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:>>F
4multi_head_attention_key_add_readvariableop_resource:>V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:>>H
6multi_head_attention_value_add_readvariableop_resource:>a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:>>O
Amulti_head_attention_attention_output_add_readvariableop_resource:>?
1layer_normalization_mul_3_readvariableop_resource:>=
/layer_normalization_add_readvariableop_resource:>9
'dense_tensordot_readvariableop_resource:>>3
%dense_biasadd_readvariableop_resource:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	6
$sex_embedding_embedding_lookup_24266:<
*age_group_embedding_embedding_lookup_24271:=
+occupation_embedding_embedding_lookup_24276:A
3layer_normalization_1_mul_3_readvariableop_resource:>?
1layer_normalization_1_add_readvariableop_resource:>:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?D
5batch_normalization_batchnorm_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_batchnorm_readvariableop_1_resource:	?F
7batch_normalization_batchnorm_readvariableop_2_resource:	?:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??$age_group_embedding/embedding_lookup?,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?genres_vector/embedding_lookup? genres_vector/embedding_lookup_1?&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_3/ReadVariableOp?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_3/ReadVariableOp? movie_embedding/embedding_lookup?"movie_embedding/embedding_lookup_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?%occupation_embedding/embedding_lookup?:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp?<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp?<process_movie_embedding_with_genres/Tensordot/ReadVariableOp?>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp?sex_embedding/embedding_lookup?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_sequence_movie_ids>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdinputs_sequence_movie_ids movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
 movie_embedding/embedding_lookupResourceGather&movie_embedding_embedding_lookup_23949$movie_index_lookup/Identity:output:0*
Tindices0	*9
_class/
-+loc:@movie_embedding/embedding_lookup/23949*+
_output_shapes
:?????????>*
dtype0?
)movie_embedding/embedding_lookup/IdentityIdentity)movie_embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@movie_embedding/embedding_lookup/23949*+
_output_shapes
:?????????>?
+movie_embedding/embedding_lookup/Identity_1Identity2movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>?
genres_vector/embedding_lookupResourceGather$genres_vector_embedding_lookup_23954$movie_index_lookup/Identity:output:0*
Tindices0	*7
_class-
+)loc:@genres_vector/embedding_lookup/23954*+
_output_shapes
:?????????*
dtype0?
'genres_vector/embedding_lookup/IdentityIdentity'genres_vector/embedding_lookup:output:0*
T0*7
_class-
+)loc:@genres_vector/embedding_lookup/23954*+
_output_shapes
:??????????
)genres_vector/embedding_lookup/Identity_1Identity0genres_vector/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_2/concatConcatV24movie_embedding/embedding_lookup/Identity_1:output:02genres_vector/embedding_lookup/Identity_1:output:0"concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
<process_movie_embedding_with_genres/Tensordot/ReadVariableOpReadVariableOpEprocess_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0|
2process_movie_embedding_with_genres/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
2process_movie_embedding_with_genres/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
3process_movie_embedding_with_genres/Tensordot/ShapeShapeconcatenate_2/concat:output:0*
T0*
_output_shapes
:}
;process_movie_embedding_with_genres/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot/GatherV2GatherV2<process_movie_embedding_with_genres/Tensordot/Shape:output:0;process_movie_embedding_with_genres/Tensordot/free:output:0Dprocess_movie_embedding_with_genres/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=process_movie_embedding_with_genres/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot/GatherV2_1GatherV2<process_movie_embedding_with_genres/Tensordot/Shape:output:0;process_movie_embedding_with_genres/Tensordot/axes:output:0Fprocess_movie_embedding_with_genres/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3process_movie_embedding_with_genres/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2process_movie_embedding_with_genres/Tensordot/ProdProd?process_movie_embedding_with_genres/Tensordot/GatherV2:output:0<process_movie_embedding_with_genres/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5process_movie_embedding_with_genres/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
4process_movie_embedding_with_genres/Tensordot/Prod_1ProdAprocess_movie_embedding_with_genres/Tensordot/GatherV2_1:output:0>process_movie_embedding_with_genres/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9process_movie_embedding_with_genres/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4process_movie_embedding_with_genres/Tensordot/concatConcatV2;process_movie_embedding_with_genres/Tensordot/free:output:0;process_movie_embedding_with_genres/Tensordot/axes:output:0Bprocess_movie_embedding_with_genres/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3process_movie_embedding_with_genres/Tensordot/stackPack;process_movie_embedding_with_genres/Tensordot/Prod:output:0=process_movie_embedding_with_genres/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
7process_movie_embedding_with_genres/Tensordot/transpose	Transposeconcatenate_2/concat:output:0=process_movie_embedding_with_genres/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
5process_movie_embedding_with_genres/Tensordot/ReshapeReshape;process_movie_embedding_with_genres/Tensordot/transpose:y:0<process_movie_embedding_with_genres/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
4process_movie_embedding_with_genres/Tensordot/MatMulMatMul>process_movie_embedding_with_genres/Tensordot/Reshape:output:0Dprocess_movie_embedding_with_genres/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>
5process_movie_embedding_with_genres/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>}
;process_movie_embedding_with_genres/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot/concat_1ConcatV2?process_movie_embedding_with_genres/Tensordot/GatherV2:output:0>process_movie_embedding_with_genres/Tensordot/Const_2:output:0Dprocess_movie_embedding_with_genres/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
-process_movie_embedding_with_genres/TensordotReshape>process_movie_embedding_with_genres/Tensordot/MatMul:product:0?process_movie_embedding_with_genres/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
:process_movie_embedding_with_genres/BiasAdd/ReadVariableOpReadVariableOpCprocess_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
+process_movie_embedding_with_genres/BiasAddBiasAdd6process_movie_embedding_with_genres/Tensordot:output:0Bprocess_movie_embedding_with_genres/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
(process_movie_embedding_with_genres/ReluRelu4process_movie_embedding_with_genres/BiasAdd:output:0*
T0*+
_output_shapes
:?????????>?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_target_movie_id>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdinputs_target_movie_id"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV26process_movie_embedding_with_genres/Relu:activations:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimsinputs_sequence_ratings&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
"movie_embedding/embedding_lookup_1ResourceGather&movie_embedding_embedding_lookup_23949&movie_index_lookup/Identity_1:output:0*
Tindices0	*9
_class/
-+loc:@movie_embedding/embedding_lookup/23949*+
_output_shapes
:?????????>*
dtype0?
+movie_embedding/embedding_lookup_1/IdentityIdentity+movie_embedding/embedding_lookup_1:output:0*
T0*9
_class/
-+loc:@movie_embedding/embedding_lookup/23949*+
_output_shapes
:?????????>?
-movie_embedding/embedding_lookup_1/Identity_1Identity4movie_embedding/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????>?
 genres_vector/embedding_lookup_1ResourceGather$genres_vector_embedding_lookup_23954&movie_index_lookup/Identity_1:output:0*
Tindices0	*7
_class-
+)loc:@genres_vector/embedding_lookup/23954*+
_output_shapes
:?????????*
dtype0?
)genres_vector/embedding_lookup_1/IdentityIdentity)genres_vector/embedding_lookup_1:output:0*
T0*7
_class-
+)loc:@genres_vector/embedding_lookup/23954*+
_output_shapes
:??????????
+genres_vector/embedding_lookup_1/Identity_1Identity2genres_vector/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:??????????
multiply/mulMultf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*+
_output_shapes
:?????????>[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_1/concatConcatV26movie_embedding/embedding_lookup_1/Identity_1:output:04genres_vector/embedding_lookup_1/Identity_1:output:0"concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
tf.unstack/unstackUnpackmultiply/mul:z:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOpReadVariableOpEprocess_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0~
4process_movie_embedding_with_genres/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4process_movie_embedding_with_genres/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5process_movie_embedding_with_genres/Tensordot_1/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:
=process_movie_embedding_with_genres/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot_1/GatherV2GatherV2>process_movie_embedding_with_genres/Tensordot_1/Shape:output:0=process_movie_embedding_with_genres/Tensordot_1/free:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?process_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:process_movie_embedding_with_genres/Tensordot_1/GatherV2_1GatherV2>process_movie_embedding_with_genres/Tensordot_1/Shape:output:0=process_movie_embedding_with_genres/Tensordot_1/axes:output:0Hprocess_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5process_movie_embedding_with_genres/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4process_movie_embedding_with_genres/Tensordot_1/ProdProdAprocess_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0>process_movie_embedding_with_genres/Tensordot_1/Const:output:0*
T0*
_output_shapes
: ?
7process_movie_embedding_with_genres/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6process_movie_embedding_with_genres/Tensordot_1/Prod_1ProdCprocess_movie_embedding_with_genres/Tensordot_1/GatherV2_1:output:0@process_movie_embedding_with_genres/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: }
;process_movie_embedding_with_genres/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6process_movie_embedding_with_genres/Tensordot_1/concatConcatV2=process_movie_embedding_with_genres/Tensordot_1/free:output:0=process_movie_embedding_with_genres/Tensordot_1/axes:output:0Dprocess_movie_embedding_with_genres/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5process_movie_embedding_with_genres/Tensordot_1/stackPack=process_movie_embedding_with_genres/Tensordot_1/Prod:output:0?process_movie_embedding_with_genres/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9process_movie_embedding_with_genres/Tensordot_1/transpose	Transposeconcatenate_1/concat:output:0?process_movie_embedding_with_genres/Tensordot_1/concat:output:0*
T0*+
_output_shapes
:?????????P?
7process_movie_embedding_with_genres/Tensordot_1/ReshapeReshape=process_movie_embedding_with_genres/Tensordot_1/transpose:y:0>process_movie_embedding_with_genres/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:???????????????????
6process_movie_embedding_with_genres/Tensordot_1/MatMulMatMul@process_movie_embedding_with_genres/Tensordot_1/Reshape:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
7process_movie_embedding_with_genres/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>
=process_movie_embedding_with_genres/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8process_movie_embedding_with_genres/Tensordot_1/concat_1ConcatV2Aprocess_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0@process_movie_embedding_with_genres/Tensordot_1/Const_2:output:0Fprocess_movie_embedding_with_genres/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/process_movie_embedding_with_genres/Tensordot_1Reshape@process_movie_embedding_with_genres/Tensordot_1/MatMul:product:0Aprocess_movie_embedding_with_genres/Tensordot_1/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOpReadVariableOpCprocess_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
-process_movie_embedding_with_genres/BiasAdd_1BiasAdd8process_movie_embedding_with_genres/Tensordot_1:output:0Dprocess_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
*process_movie_embedding_with_genres/Relu_1Relu6process_movie_embedding_with_genres/BiasAdd_1:output:0*
T0*+
_output_shapes
:?????????>[
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?

concatenate_3/concatConcatV2$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:08process_movie_embedding_with_genres/Relu_1:activations:0"concatenate_3/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????>?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumconcatenate_3/concat:output:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumconcatenate_3/concat:output:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumconcatenate_3/concat:output:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>_
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:??????????
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????>?
add/addAddV2concatenate_3/concat:output:0dropout/Identity:output:0*
T0*+
_output_shapes
:?????????>T
layer_normalization/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization/ReshapeReshapeadd/add:z:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>t
layer_normalization/ones/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????u
 layer_normalization/zeros/packedPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*+
_output_shapes
:?????????>?
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>|
leaky_re_lu/LeakyRelu	LeakyRelulayer_normalization/add:z:0*+
_output_shapes
:?????????>*
alpha%???>?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:>>*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
dense/Tensordot/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose#leaky_re_lu/LeakyRelu:activations:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????>?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_occupation;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNdinputs_occupationstring_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0#^movie_index_lookup/Assert_1/Assert*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_age_group;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNdinputs_age_groupstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handle
inputs_sex9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNd
inputs_sexstring_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:?????????l
dropout_1/IdentityIdentitydense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????>?
sex_embedding/embedding_lookupResourceGather$sex_embedding_embedding_lookup_24266string_lookup/Identity:output:0*
Tindices0	*7
_class-
+)loc:@sex_embedding/embedding_lookup/24266*+
_output_shapes
:?????????*
dtype0?
'sex_embedding/embedding_lookup/IdentityIdentity'sex_embedding/embedding_lookup:output:0*
T0*7
_class-
+)loc:@sex_embedding/embedding_lookup/24266*+
_output_shapes
:??????????
)sex_embedding/embedding_lookup/Identity_1Identity0sex_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
$age_group_embedding/embedding_lookupResourceGather*age_group_embedding_embedding_lookup_24271!string_lookup_1/Identity:output:0*
Tindices0	*=
_class3
1/loc:@age_group_embedding/embedding_lookup/24271*+
_output_shapes
:?????????*
dtype0?
-age_group_embedding/embedding_lookup/IdentityIdentity-age_group_embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@age_group_embedding/embedding_lookup/24271*+
_output_shapes
:??????????
/age_group_embedding/embedding_lookup/Identity_1Identity6age_group_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
%occupation_embedding/embedding_lookupResourceGather+occupation_embedding_embedding_lookup_24276!string_lookup_2/Identity:output:0*
Tindices0	*>
_class4
20loc:@occupation_embedding/embedding_lookup/24276*+
_output_shapes
:?????????*
dtype0?
.occupation_embedding/embedding_lookup/IdentityIdentity.occupation_embedding/embedding_lookup:output:0*
T0*>
_class4
20loc:@occupation_embedding/embedding_lookup/24276*+
_output_shapes
:??????????
0occupation_embedding/embedding_lookup/Identity_1Identity7occupation_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
	add_1/addAddV2layer_normalization/add:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????>Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV22sex_embedding/embedding_lookup/Identity_1:output:08age_group_embedding/embedding_lookup/Identity_1:output:09occupation_embedding/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????X
layer_normalization_1/ShapeShapeadd_1/add:z:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_1/ReshapeReshapeadd_1/add:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>x
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:?????????y
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:?????????>?
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  ?
flatten/ReshapeReshapelayer_normalization_1/add:z:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????X
reshape/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapeconcatenate/concat:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_4/concatConcatV2flatten/Reshape:output:0reshape/Reshape:output:0"concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_1/MatMulMatMulconcatenate_4/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/mul_1Muldense_1/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
leaky_re_lu_1/LeakyRelu	LeakyRelu'batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>x
dropout_2/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/mul_1Muldense_2/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
leaky_re_lu_2/LeakyRelu	LeakyRelu)batch_normalization_1/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>x
dropout_3/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMuldropout_3/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^age_group_embedding/embedding_lookup-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^genres_vector/embedding_lookup!^genres_vector/embedding_lookup_1'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_3/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_3/ReadVariableOp!^movie_embedding/embedding_lookup#^movie_embedding/embedding_lookup_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV29^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp&^occupation_embedding/embedding_lookup;^process_movie_embedding_with_genres/BiasAdd/ReadVariableOp=^process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp=^process_movie_embedding_with_genres/Tensordot/ReadVariableOp?^process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp^sex_embedding/embedding_lookup^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$age_group_embedding/embedding_lookup$age_group_embedding/embedding_lookup2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
genres_vector/embedding_lookupgenres_vector/embedding_lookup2D
 genres_vector/embedding_lookup_1 genres_vector/embedding_lookup_12P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_3/ReadVariableOp(layer_normalization/mul_3/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_3/ReadVariableOp*layer_normalization_1/mul_3/ReadVariableOp2D
 movie_embedding/embedding_lookup movie_embedding/embedding_lookup2H
"movie_embedding/embedding_lookup_1"movie_embedding/embedding_lookup_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2N
%occupation_embedding/embedding_lookup%occupation_embedding/embedding_lookup2x
:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp:process_movie_embedding_with_genres/BiasAdd/ReadVariableOp2|
<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp<process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp2|
<process_movie_embedding_with_genres/Tensordot/ReadVariableOp<process_movie_embedding_with_genres/Tensordot/ReadVariableOp2?
>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp>process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp2@
sex_embedding/embedding_lookupsex_embedding/embedding_lookup2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/age_group:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/sequence_movie_ids:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/sequence_ratings:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/target_movie_id:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?*
?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25253	
query	
valueA
+query_einsum_einsum_readvariableop_resource:>>3
!query_add_readvariableop_resource:>?
)key_einsum_einsum_readvariableop_resource:>>1
key_add_readvariableop_resource:>A
+value_einsum_einsum_readvariableop_resource:>>3
!value_add_readvariableop_resource:>L
6attention_output_einsum_einsum_readvariableop_resource:>>:
,attention_output_add_readvariableop_resource:>
identity??#attention_output/add/ReadVariableOp?-attention_output/einsum/Einsum/ReadVariableOp?key/add/ReadVariableOp? key/einsum/Einsum/ReadVariableOp?query/add/ReadVariableOp?"query/einsum/Einsum/ReadVariableOp?value/add/ReadVariableOp?"value/einsum/Einsum/ReadVariableOp?
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:?????????q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:?????????>?
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?
`
'__inference_dropout_layer_call_fn_25305

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_22331s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_22275

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_25409

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21723s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_25526

inputs	(
embedding_lookup_25520:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_25520inputs*
Tindices0	*)
_class
loc:@embedding_lookup/25520*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/25520*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_25762

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?1
?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25295	
query	
valueA
+query_einsum_einsum_readvariableop_resource:>>3
!query_add_readvariableop_resource:>?
)key_einsum_einsum_readvariableop_resource:>>1
key_add_readvariableop_resource:>A
+value_einsum_einsum_readvariableop_resource:>>3
!value_add_readvariableop_resource:>L
6attention_output_einsum_einsum_readvariableop_resource:>>:
,attention_output_add_readvariableop_resource:>
identity??#attention_output/add/ReadVariableOp?-attention_output/einsum/Einsum/ReadVariableOp?key/add/ReadVariableOp? key/einsum/Einsum/ReadVariableOp?query/add/ReadVariableOp?"query/einsum/Einsum/ReadVariableOp?value/add/ReadVariableOp?"value/einsum/Einsum/ReadVariableOp?
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:?????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????^
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:??????????
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:?????????>?
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_25929

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680

inputs+
mul_3_readvariableop_resource:>)
add_readvariableop_resource:>
identity??add/ReadVariableOp?mul_3/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
mul_1Mulmul:z:0strided_slice_1:output:0*
T0*
_output_shapes
: _
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_2Mulmul_2/x:output:0strided_slice_2:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0	mul_1:z:0	mul_2:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????>L
ones/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????M
zeros/packedPack	mul_1:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:p
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*+
_output_shapes
:?????????>n
mul_3/ReadVariableOpReadVariableOpmul_3_readvariableop_resource*
_output_shapes
:>*
dtype0t
mul_3MulReshape_1:output:0mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:>*
dtype0i
addAddV2	mul_3:z:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????>r
NoOpNoOp^add/ReadVariableOp^mul_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_3/ReadVariableOpmul_3/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807

inputs	(
embedding_lookup_21801:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21801inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21801*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21801*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_25439

inputs3
!tensordot_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:>>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????>?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?	
^
B__inference_reshape_layer_call_and_return_conditional_losses_21902

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_21600	
query	
valueA
+query_einsum_einsum_readvariableop_resource:>>3
!query_add_readvariableop_resource:>?
)key_einsum_einsum_readvariableop_resource:>>1
key_add_readvariableop_resource:>A
+value_einsum_einsum_readvariableop_resource:>>3
!value_add_readvariableop_resource:>L
6attention_output_einsum_einsum_readvariableop_resource:>>:
,attention_output_add_readvariableop_resource:>
identity??#attention_output/add/ReadVariableOp?-attention_output/einsum/Einsum/ReadVariableOp?key/add/ReadVariableOp? key/einsum/Einsum/ReadVariableOp?query/add/ReadVariableOp?"query/einsum/Einsum/ReadVariableOp?value/add/ReadVariableOp?"value/einsum/Einsum/ReadVariableOp?
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>c
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbel
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:?????????q
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>k
IdentityIdentityattention_output/add:z:0^NoOp*
T0*+
_output_shapes
:?????????>?
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????>*
alpha%???>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
l
@__inference_add_1_layer_call_and_return_conditional_losses_25478
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1
?
Y
-__inference_concatenate_4_layer_call_fn_25631
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????:?????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
__inference_<lambda>_260096
2key_value_init146_lookuptableimportv2_table_handle.
*key_value_init146_lookuptableimportv2_keys0
,key_value_init146_lookuptableimportv2_values	
identity??%key_value_init146/LookupTableImportV2?
%key_value_init146/LookupTableImportV2LookupTableImportV22key_value_init146_lookuptableimportv2_table_handle*key_value_init146_lookuptableimportv2_keys,key_value_init146_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init146/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init146/LookupTableImportV2%key_value_init146/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25703

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_leaky_re_lu_2_layer_call_fn_25878

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24955

inputs	)
embedding_lookup_24949:	?>
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_24949inputs*
Tindices0	*)
_class
loc:@embedding_lookup/24949*+
_output_shapes
:?????????>*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/24949*+
_output_shapes
:?????????>?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????>Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_22099
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_id
unknown
	unknown_0	
	unknown_1:	?>
	unknown_2:	?
	unknown_3:P>
	unknown_4:>
	unknown_5
	unknown_6:>>
	unknown_7:>
	unknown_8:>>
	unknown_9:> 

unknown_10:>>

unknown_11:> 

unknown_12:>>

unknown_13:>

unknown_14:>

unknown_15:>

unknown_16:>>

unknown_17:>

unknown_18

unknown_19	

unknown_20

unknown_21	

unknown_22

unknown_23	

unknown_24:

unknown_25:

unknown_26:

unknown_27:>

unknown_28:>

unknown_29:
??

unknown_30:	?

unknown_31:	?

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*E
_read_only_resource_inputs'
%#	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_22008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
C
'__inference_flatten_layer_call_fn_25602

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_21888a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?"
?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23
	inputs_24
	inputs_25
	inputs_26
	inputs_27
	inputs_28
	inputs_29
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????>[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:S	O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:S
O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?%
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21240

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	??
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype0?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:??
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_1_layer_call_fn_25098
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_25400

inputs
identity[
	LeakyRelu	LeakyReluinputs*+
_output_shapes
:?????????>*
alpha%???>c
IdentityIdentityLeakyRelu:activations:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_23928
inputs_age_group
inputs_occupation
inputs_sequence_movie_ids
inputs_sequence_ratings

inputs_sex
inputs_target_movie_id
inputs_user_id
unknown
	unknown_0	
	unknown_1:	?>
	unknown_2:	?
	unknown_3:P>
	unknown_4:>
	unknown_5
	unknown_6:>>
	unknown_7:>
	unknown_8:>>
	unknown_9:> 

unknown_10:>>

unknown_11:> 

unknown_12:>>

unknown_13:>

unknown_14:>

unknown_15:>

unknown_16:>>

unknown_17:>

unknown_18

unknown_19	

unknown_20

unknown_21	

unknown_22

unknown_23	

unknown_24:

unknown_25:

unknown_26:

unknown_27:>

unknown_28:>

unknown_29:
??

unknown_30:	?

unknown_31:	?

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_age_groupinputs_occupationinputs_sequence_movie_idsinputs_sequence_ratings
inputs_sexinputs_target_movie_idinputs_user_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*A
_read_only_resource_inputs#
!	
 !"#$%&)*+,/012*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_22901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/age_group:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/sequence_movie_ids:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/sequence_ratings:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/target_movie_id:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_25608

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_25806

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21193p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_25774

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_260335
1key_value_init97_lookuptableimportv2_table_handle-
)key_value_init97_lookuptableimportv2_keys/
+key_value_init97_lookuptableimportv2_values	
identity??$key_value_init97/LookupTableImportV2?
$key_value_init97/LookupTableImportV2LookupTableImportV21key_value_init97_lookuptableimportv2_table_handle)key_value_init97_lookuptableimportv2_keys+key_value_init97_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: m
NoOpNoOp%^key_value_init97/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$key_value_init97/LookupTableImportV2$key_value_init97/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
/__inference_movie_embedding_layer_call_fn_24930

inputs	
unknown:	?>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25080

inputs3
!tensordot_readvariableop_resource:P>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????>e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_21923

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_21723

inputs3
!tensordot_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:>>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????>?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?.
 __inference__wrapped_model_21087
	age_group

occupation
sequence_movie_ids
sequence_ratings
sex
target_movie_id
user_idG
Cmodel_movie_index_lookup_none_lookup_lookuptablefindv2_table_handleH
Dmodel_movie_index_lookup_none_lookup_lookuptablefindv2_default_value	?
,model_movie_embedding_embedding_lookup_20642:	?>=
*model_genres_vector_embedding_lookup_20647:	?]
Kmodel_process_movie_embedding_with_genres_tensordot_readvariableop_resource:P>W
Imodel_process_movie_embedding_with_genres_biasadd_readvariableop_resource:>&
"model_tf___operators___add_addv2_y\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:>>N
<model_multi_head_attention_query_add_readvariableop_resource:>Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:>>L
:model_multi_head_attention_key_add_readvariableop_resource:>\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:>>N
<model_multi_head_attention_value_add_readvariableop_resource:>g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:>>U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:>E
7model_layer_normalization_mul_3_readvariableop_resource:>C
5model_layer_normalization_add_readvariableop_resource:>?
-model_dense_tensordot_readvariableop_resource:>>9
+model_dense_biasadd_readvariableop_resource:>D
@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value	B
>model_string_lookup_none_lookup_lookuptablefindv2_table_handleC
?model_string_lookup_none_lookup_lookuptablefindv2_default_value	<
*model_sex_embedding_embedding_lookup_20959:B
0model_age_group_embedding_embedding_lookup_20964:C
1model_occupation_embedding_embedding_lookup_20969:G
9model_layer_normalization_1_mul_3_readvariableop_resource:>E
7model_layer_normalization_1_add_readvariableop_resource:>@
,model_dense_1_matmul_readvariableop_resource:
??<
-model_dense_1_biasadd_readvariableop_resource:	?J
;model_batch_normalization_batchnorm_readvariableop_resource:	?N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	?L
=model_batch_normalization_batchnorm_readvariableop_1_resource:	?L
=model_batch_normalization_batchnorm_readvariableop_2_resource:	?@
,model_dense_2_matmul_readvariableop_resource:
??<
-model_dense_2_biasadd_readvariableop_resource:	?L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	?P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	?N
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	?N
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	??
,model_dense_3_matmul_readvariableop_resource:	?;
-model_dense_3_biasadd_readvariableop_resource:
identity??*model/age_group_embedding/embedding_lookup?2model/batch_normalization/batchnorm/ReadVariableOp?4model/batch_normalization/batchnorm/ReadVariableOp_1?4model/batch_normalization/batchnorm/ReadVariableOp_2?6model/batch_normalization/batchnorm/mul/ReadVariableOp?4model/batch_normalization_1/batchnorm/ReadVariableOp?6model/batch_normalization_1/batchnorm/ReadVariableOp_1?6model/batch_normalization_1/batchnorm/ReadVariableOp_2?8model/batch_normalization_1/batchnorm/mul/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/Tensordot/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?$model/dense_3/BiasAdd/ReadVariableOp?#model/dense_3/MatMul/ReadVariableOp?$model/genres_vector/embedding_lookup?&model/genres_vector/embedding_lookup_1?,model/layer_normalization/add/ReadVariableOp?.model/layer_normalization/mul_3/ReadVariableOp?.model/layer_normalization_1/add/ReadVariableOp?0model/layer_normalization_1/mul_3/ReadVariableOp?&model/movie_embedding/embedding_lookup?(model/movie_embedding/embedding_lookup_1?&model/movie_index_lookup/Assert/Assert?(model/movie_index_lookup/Assert_1/Assert?6model/movie_index_lookup/None_Lookup/LookupTableFindV2?8model/movie_index_lookup/None_Lookup_1/LookupTableFindV2?>model/multi_head_attention/attention_output/add/ReadVariableOp?Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?1model/multi_head_attention/key/add/ReadVariableOp?;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp?3model/multi_head_attention/query/add/ReadVariableOp?=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp?3model/multi_head_attention/value/add/ReadVariableOp?=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp?+model/occupation_embedding/embedding_lookup?@model/process_movie_embedding_with_genres/BiasAdd/ReadVariableOp?Bmodel/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp?Bmodel/process_movie_embedding_with_genres/Tensordot/ReadVariableOp?Dmodel/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp?$model/sex_embedding/embedding_lookup?!model/string_lookup/Assert/Assert?1model/string_lookup/None_Lookup/LookupTableFindV2?#model/string_lookup_1/Assert/Assert?3model/string_lookup_1/None_Lookup/LookupTableFindV2?#model/string_lookup_2/Assert/Assert?3model/string_lookup_2/None_Lookup/LookupTableFindV2?
6model/movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Cmodel_movie_index_lookup_none_lookup_lookuptablefindv2_table_handlesequence_movie_idsDmodel_movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????k
 model/movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
model/movie_index_lookup/EqualEqual?model/movie_index_lookup/None_Lookup/LookupTableFindV2:values:0)model/movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????t
model/movie_index_lookup/WhereWhere"model/movie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
!model/movie_index_lookup/GatherNdGatherNdsequence_movie_ids&model/movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
%model/movie_index_lookup/StringFormatStringFormat*model/movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.n
model/movie_index_lookup/SizeSize&model/movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: d
"model/movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
 model/movie_index_lookup/Equal_1Equal&model/movie_index_lookup/Size:output:0+model/movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
&model/movie_index_lookup/Assert/AssertAssert$model/movie_index_lookup/Equal_1:z:0.model/movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
!model/movie_index_lookup/IdentityIdentity?model/movie_index_lookup/None_Lookup/LookupTableFindV2:values:0'^model/movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
&model/movie_embedding/embedding_lookupResourceGather,model_movie_embedding_embedding_lookup_20642*model/movie_index_lookup/Identity:output:0*
Tindices0	*?
_class5
31loc:@model/movie_embedding/embedding_lookup/20642*+
_output_shapes
:?????????>*
dtype0?
/model/movie_embedding/embedding_lookup/IdentityIdentity/model/movie_embedding/embedding_lookup:output:0*
T0*?
_class5
31loc:@model/movie_embedding/embedding_lookup/20642*+
_output_shapes
:?????????>?
1model/movie_embedding/embedding_lookup/Identity_1Identity8model/movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>?
$model/genres_vector/embedding_lookupResourceGather*model_genres_vector_embedding_lookup_20647*model/movie_index_lookup/Identity:output:0*
Tindices0	*=
_class3
1/loc:@model/genres_vector/embedding_lookup/20647*+
_output_shapes
:?????????*
dtype0?
-model/genres_vector/embedding_lookup/IdentityIdentity-model/genres_vector/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model/genres_vector/embedding_lookup/20647*+
_output_shapes
:??????????
/model/genres_vector/embedding_lookup/Identity_1Identity6model/genres_vector/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_2/concatConcatV2:model/movie_embedding/embedding_lookup/Identity_1:output:08model/genres_vector/embedding_lookup/Identity_1:output:0(model/concatenate_2/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
Bmodel/process_movie_embedding_with_genres/Tensordot/ReadVariableOpReadVariableOpKmodel_process_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0?
8model/process_movie_embedding_with_genres/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
8model/process_movie_embedding_with_genres/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
9model/process_movie_embedding_with_genres/Tensordot/ShapeShape#model/concatenate_2/concat:output:0*
T0*
_output_shapes
:?
Amodel/process_movie_embedding_with_genres/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<model/process_movie_embedding_with_genres/Tensordot/GatherV2GatherV2Bmodel/process_movie_embedding_with_genres/Tensordot/Shape:output:0Amodel/process_movie_embedding_with_genres/Tensordot/free:output:0Jmodel/process_movie_embedding_with_genres/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Cmodel/process_movie_embedding_with_genres/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>model/process_movie_embedding_with_genres/Tensordot/GatherV2_1GatherV2Bmodel/process_movie_embedding_with_genres/Tensordot/Shape:output:0Amodel/process_movie_embedding_with_genres/Tensordot/axes:output:0Lmodel/process_movie_embedding_with_genres/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
9model/process_movie_embedding_with_genres/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
8model/process_movie_embedding_with_genres/Tensordot/ProdProdEmodel/process_movie_embedding_with_genres/Tensordot/GatherV2:output:0Bmodel/process_movie_embedding_with_genres/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
;model/process_movie_embedding_with_genres/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
:model/process_movie_embedding_with_genres/Tensordot/Prod_1ProdGmodel/process_movie_embedding_with_genres/Tensordot/GatherV2_1:output:0Dmodel/process_movie_embedding_with_genres/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ?
?model/process_movie_embedding_with_genres/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model/process_movie_embedding_with_genres/Tensordot/concatConcatV2Amodel/process_movie_embedding_with_genres/Tensordot/free:output:0Amodel/process_movie_embedding_with_genres/Tensordot/axes:output:0Hmodel/process_movie_embedding_with_genres/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
9model/process_movie_embedding_with_genres/Tensordot/stackPackAmodel/process_movie_embedding_with_genres/Tensordot/Prod:output:0Cmodel/process_movie_embedding_with_genres/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
=model/process_movie_embedding_with_genres/Tensordot/transpose	Transpose#model/concatenate_2/concat:output:0Cmodel/process_movie_embedding_with_genres/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
;model/process_movie_embedding_with_genres/Tensordot/ReshapeReshapeAmodel/process_movie_embedding_with_genres/Tensordot/transpose:y:0Bmodel/process_movie_embedding_with_genres/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
:model/process_movie_embedding_with_genres/Tensordot/MatMulMatMulDmodel/process_movie_embedding_with_genres/Tensordot/Reshape:output:0Jmodel/process_movie_embedding_with_genres/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
;model/process_movie_embedding_with_genres/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>?
Amodel/process_movie_embedding_with_genres/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<model/process_movie_embedding_with_genres/Tensordot/concat_1ConcatV2Emodel/process_movie_embedding_with_genres/Tensordot/GatherV2:output:0Dmodel/process_movie_embedding_with_genres/Tensordot/Const_2:output:0Jmodel/process_movie_embedding_with_genres/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
3model/process_movie_embedding_with_genres/TensordotReshapeDmodel/process_movie_embedding_with_genres/Tensordot/MatMul:product:0Emodel/process_movie_embedding_with_genres/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
@model/process_movie_embedding_with_genres/BiasAdd/ReadVariableOpReadVariableOpImodel_process_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
1model/process_movie_embedding_with_genres/BiasAddBiasAdd<model/process_movie_embedding_with_genres/Tensordot:output:0Hmodel/process_movie_embedding_with_genres/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
.model/process_movie_embedding_with_genres/ReluRelu:model/process_movie_embedding_with_genres/BiasAdd:output:0*
T0*+
_output_shapes
:?????????>?
8model/movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2Cmodel_movie_index_lookup_none_lookup_lookuptablefindv2_table_handletarget_movie_idDmodel_movie_index_lookup_none_lookup_lookuptablefindv2_default_value7^model/movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????m
"model/movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
 model/movie_index_lookup/Equal_2EqualAmodel/movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0+model/movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????x
 model/movie_index_lookup/Where_1Where$model/movie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
#model/movie_index_lookup/GatherNd_1GatherNdtarget_movie_id(model/movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
'model/movie_index_lookup/StringFormat_1StringFormat,model/movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model/movie_index_lookup/Size_1Size(model/movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: d
"model/movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
 model/movie_index_lookup/Equal_3Equal(model/movie_index_lookup/Size_1:output:0+model/movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
(model/movie_index_lookup/Assert_1/AssertAssert$model/movie_index_lookup/Equal_3:z:00model/movie_index_lookup/StringFormat_1:output:0'^model/movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
#model/movie_index_lookup/Identity_1IdentityAmodel/movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0)^model/movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
 model/tf.__operators__.add/AddV2AddV2<model/process_movie_embedding_with_genres/Relu:activations:0"model_tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>n
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
model/tf.expand_dims/ExpandDims
ExpandDimssequence_ratings,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
(model/movie_embedding/embedding_lookup_1ResourceGather,model_movie_embedding_embedding_lookup_20642,model/movie_index_lookup/Identity_1:output:0*
Tindices0	*?
_class5
31loc:@model/movie_embedding/embedding_lookup/20642*+
_output_shapes
:?????????>*
dtype0?
1model/movie_embedding/embedding_lookup_1/IdentityIdentity1model/movie_embedding/embedding_lookup_1:output:0*
T0*?
_class5
31loc:@model/movie_embedding/embedding_lookup/20642*+
_output_shapes
:?????????>?
3model/movie_embedding/embedding_lookup_1/Identity_1Identity:model/movie_embedding/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:?????????>?
&model/genres_vector/embedding_lookup_1ResourceGather*model_genres_vector_embedding_lookup_20647,model/movie_index_lookup/Identity_1:output:0*
Tindices0	*=
_class3
1/loc:@model/genres_vector/embedding_lookup/20647*+
_output_shapes
:?????????*
dtype0?
/model/genres_vector/embedding_lookup_1/IdentityIdentity/model/genres_vector/embedding_lookup_1:output:0*
T0*=
_class3
1/loc:@model/genres_vector/embedding_lookup/20647*+
_output_shapes
:??????????
1model/genres_vector/embedding_lookup_1/Identity_1Identity8model/genres_vector/embedding_lookup_1/Identity:output:0*
T0*+
_output_shapes
:??????????
model/multiply/mulMul$model/tf.__operators__.add/AddV2:z:0(model/tf.expand_dims/ExpandDims:output:0*
T0*+
_output_shapes
:?????????>a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_1/concatConcatV2<model/movie_embedding/embedding_lookup_1/Identity_1:output:0:model/genres_vector/embedding_lookup_1/Identity_1:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P?
model/tf.unstack/unstackUnpackmodel/multiply/mul:z:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numg
%model/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_1/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:0.model/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_2/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:1.model/tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_3/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:2.model/tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_4/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:3.model/tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_5/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:4.model/tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_6/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:5.model/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_7/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:6.model/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_8/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:7.model/tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>g
%model/tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
!model/tf.expand_dims_9/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:8.model/tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_10/ExpandDims
ExpandDims!model/tf.unstack/unstack:output:9/model/tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_11/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:10/model/tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_12/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:11/model/tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_13/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:12/model/tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_14/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:13/model/tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_15/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:14/model/tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_16/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:15/model/tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_17/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:16/model/tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_18/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:17/model/tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_19/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:18/model/tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_20/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:19/model/tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_21/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:20/model/tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_22/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:21/model/tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_23/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:22/model/tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_24/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:23/model/tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_25/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:24/model/tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_26/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:25/model/tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_27/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:26/model/tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_28/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:27/model/tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>h
&model/tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
"model/tf.expand_dims_29/ExpandDims
ExpandDims"model/tf.unstack/unstack:output:28/model/tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
Dmodel/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOpReadVariableOpKmodel_process_movie_embedding_with_genres_tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0?
:model/process_movie_embedding_with_genres/Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:?
:model/process_movie_embedding_with_genres/Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
;model/process_movie_embedding_with_genres/Tensordot_1/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
:?
Cmodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>model/process_movie_embedding_with_genres/Tensordot_1/GatherV2GatherV2Dmodel/process_movie_embedding_with_genres/Tensordot_1/Shape:output:0Cmodel/process_movie_embedding_with_genres/Tensordot_1/free:output:0Lmodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Emodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
@model/process_movie_embedding_with_genres/Tensordot_1/GatherV2_1GatherV2Dmodel/process_movie_embedding_with_genres/Tensordot_1/Shape:output:0Cmodel/process_movie_embedding_with_genres/Tensordot_1/axes:output:0Nmodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
;model/process_movie_embedding_with_genres/Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
:model/process_movie_embedding_with_genres/Tensordot_1/ProdProdGmodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0Dmodel/process_movie_embedding_with_genres/Tensordot_1/Const:output:0*
T0*
_output_shapes
: ?
=model/process_movie_embedding_with_genres/Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
<model/process_movie_embedding_with_genres/Tensordot_1/Prod_1ProdImodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2_1:output:0Fmodel/process_movie_embedding_with_genres/Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: ?
Amodel/process_movie_embedding_with_genres/Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<model/process_movie_embedding_with_genres/Tensordot_1/concatConcatV2Cmodel/process_movie_embedding_with_genres/Tensordot_1/free:output:0Cmodel/process_movie_embedding_with_genres/Tensordot_1/axes:output:0Jmodel/process_movie_embedding_with_genres/Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;model/process_movie_embedding_with_genres/Tensordot_1/stackPackCmodel/process_movie_embedding_with_genres/Tensordot_1/Prod:output:0Emodel/process_movie_embedding_with_genres/Tensordot_1/Prod_1:output:0*
N*
T0*
_output_shapes
:?
?model/process_movie_embedding_with_genres/Tensordot_1/transpose	Transpose#model/concatenate_1/concat:output:0Emodel/process_movie_embedding_with_genres/Tensordot_1/concat:output:0*
T0*+
_output_shapes
:?????????P?
=model/process_movie_embedding_with_genres/Tensordot_1/ReshapeReshapeCmodel/process_movie_embedding_with_genres/Tensordot_1/transpose:y:0Dmodel/process_movie_embedding_with_genres/Tensordot_1/stack:output:0*
T0*0
_output_shapes
:???????????????????
<model/process_movie_embedding_with_genres/Tensordot_1/MatMulMatMulFmodel/process_movie_embedding_with_genres/Tensordot_1/Reshape:output:0Lmodel/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>?
=model/process_movie_embedding_with_genres/Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>?
Cmodel/process_movie_embedding_with_genres/Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
>model/process_movie_embedding_with_genres/Tensordot_1/concat_1ConcatV2Gmodel/process_movie_embedding_with_genres/Tensordot_1/GatherV2:output:0Fmodel/process_movie_embedding_with_genres/Tensordot_1/Const_2:output:0Lmodel/process_movie_embedding_with_genres/Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
5model/process_movie_embedding_with_genres/Tensordot_1ReshapeFmodel/process_movie_embedding_with_genres/Tensordot_1/MatMul:product:0Gmodel/process_movie_embedding_with_genres/Tensordot_1/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
Bmodel/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOpReadVariableOpImodel_process_movie_embedding_with_genres_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
3model/process_movie_embedding_with_genres/BiasAdd_1BiasAdd>model/process_movie_embedding_with_genres/Tensordot_1:output:0Jmodel/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
0model/process_movie_embedding_with_genres/Relu_1Relu<model/process_movie_embedding_with_genres/BiasAdd_1:output:0*
T0*+
_output_shapes
:?????????>a
model/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_3/concatConcatV2*model/tf.expand_dims_1/ExpandDims:output:0*model/tf.expand_dims_2/ExpandDims:output:0*model/tf.expand_dims_3/ExpandDims:output:0*model/tf.expand_dims_4/ExpandDims:output:0*model/tf.expand_dims_5/ExpandDims:output:0*model/tf.expand_dims_6/ExpandDims:output:0*model/tf.expand_dims_7/ExpandDims:output:0*model/tf.expand_dims_8/ExpandDims:output:0*model/tf.expand_dims_9/ExpandDims:output:0+model/tf.expand_dims_10/ExpandDims:output:0+model/tf.expand_dims_11/ExpandDims:output:0+model/tf.expand_dims_12/ExpandDims:output:0+model/tf.expand_dims_13/ExpandDims:output:0+model/tf.expand_dims_14/ExpandDims:output:0+model/tf.expand_dims_15/ExpandDims:output:0+model/tf.expand_dims_16/ExpandDims:output:0+model/tf.expand_dims_17/ExpandDims:output:0+model/tf.expand_dims_18/ExpandDims:output:0+model/tf.expand_dims_19/ExpandDims:output:0+model/tf.expand_dims_20/ExpandDims:output:0+model/tf.expand_dims_21/ExpandDims:output:0+model/tf.expand_dims_22/ExpandDims:output:0+model/tf.expand_dims_23/ExpandDims:output:0+model/tf.expand_dims_24/ExpandDims:output:0+model/tf.expand_dims_25/ExpandDims:output:0+model/tf.expand_dims_26/ExpandDims:output:0+model/tf.expand_dims_27/ExpandDims:output:0+model/tf.expand_dims_28/ExpandDims:output:0+model/tf.expand_dims_29/ExpandDims:output:0>model/process_movie_embedding_with_genres/Relu_1:activations:0(model/concatenate_3/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????>?
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
.model/multi_head_attention/query/einsum/EinsumEinsum#model/concatenate_3/concat:output:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
$model/multi_head_attention/query/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
,model/multi_head_attention/key/einsum/EinsumEinsum#model/concatenate_3/concat:output:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
"model/multi_head_attention/key/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>?
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
.model/multi_head_attention/value/einsum/EinsumEinsum#model/concatenate_3/concat:output:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:?????????>*
equationabc,cde->abde?
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:>*
dtype0?
$model/multi_head_attention/value/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????>e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *R>?
model/multi_head_attention/MulMul(model/multi_head_attention/query/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:?????????>?
(model/multi_head_attention/einsum/EinsumEinsum&model/multi_head_attention/key/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????*
equationaecd,abcd->acbe?
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:??????????
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:??????????
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0(model/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:?????????>*
equationacbe,aecd->abcd?
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:>>*
dtype0?
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????>*
equationabcd,cde->abe?
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
/model/multi_head_attention/attention_output/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
model/dropout/IdentityIdentity3model/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????>?
model/add/addAddV2#model/concatenate_3/concat:output:0model/dropout/Identity:output:0*
T0*+
_output_shapes
:?????????>`
model/layer_normalization/ShapeShapemodel/add/add:z:0*
T0*
_output_shapes
:w
-model/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'model/layer_normalization/strided_sliceStridedSlice(model/layer_normalization/Shape:output:06model/layer_normalization/strided_slice/stack:output:08model/layer_normalization/strided_slice/stack_1:output:08model/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
model/layer_normalization/mulMul(model/layer_normalization/mul/x:output:00model/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model/layer_normalization/strided_slice_1StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_1/stack:output:0:model/layer_normalization/strided_slice_1/stack_1:output:0:model/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
model/layer_normalization/mul_1Mul!model/layer_normalization/mul:z:02model/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: y
/model/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model/layer_normalization/strided_slice_2StridedSlice(model/layer_normalization/Shape:output:08model/layer_normalization/strided_slice_2/stack:output:0:model/layer_normalization/strided_slice_2/stack_1:output:0:model/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
model/layer_normalization/mul_2Mul*model/layer_normalization/mul_2/x:output:02model/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: k
)model/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :k
)model/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
'model/layer_normalization/Reshape/shapePack2model/layer_normalization/Reshape/shape/0:output:0#model/layer_normalization/mul_1:z:0#model/layer_normalization/mul_2:z:02model/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
!model/layer_normalization/ReshapeReshapemodel/add/add:z:00model/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>?
%model/layer_normalization/ones/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:i
$model/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
model/layer_normalization/onesFill.model/layer_normalization/ones/packed:output:0-model/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:??????????
&model/layer_normalization/zeros/packedPack#model/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:j
%model/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
model/layer_normalization/zerosFill/model/layer_normalization/zeros/packed:output:0.model/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????b
model/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!model/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
*model/layer_normalization/FusedBatchNormV3FusedBatchNormV3*model/layer_normalization/Reshape:output:0'model/layer_normalization/ones:output:0(model/layer_normalization/zeros:output:0(model/layer_normalization/Const:output:0*model/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
#model/layer_normalization/Reshape_1Reshape.model/layer_normalization/FusedBatchNormV3:y:0(model/layer_normalization/Shape:output:0*
T0*+
_output_shapes
:?????????>?
.model/layer_normalization/mul_3/ReadVariableOpReadVariableOp7model_layer_normalization_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
model/layer_normalization/mul_3Mul,model/layer_normalization/Reshape_1:output:06model/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
,model/layer_normalization/add/ReadVariableOpReadVariableOp5model_layer_normalization_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
model/layer_normalization/addAddV2#model/layer_normalization/mul_3:z:04model/layer_normalization/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
model/leaky_re_lu/LeakyRelu	LeakyRelu!model/layer_normalization/add:z:0*+
_output_shapes
:?????????>*
alpha%???>?
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:>>*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       t
model/dense/Tensordot/ShapeShape)model/leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
model/dense/Tensordot/transpose	Transpose)model/leaky_re_lu/LeakyRelu:activations:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????>?
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
3model/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handle
occupationAmodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????h
model/string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
model/string_lookup_2/EqualEqual<model/string_lookup_2/None_Lookup/LookupTableFindV2:values:0&model/string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????n
model/string_lookup_2/WhereWheremodel/string_lookup_2/Equal:z:0*'
_output_shapes
:??????????
model/string_lookup_2/GatherNdGatherNd
occupation#model/string_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
"model/string_lookup_2/StringFormatStringFormat'model/string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.h
model/string_lookup_2/SizeSize#model/string_lookup_2/Where:index:0*
T0	*
_output_shapes
: a
model/string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
model/string_lookup_2/Equal_1Equal#model/string_lookup_2/Size:output:0(model/string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
#model/string_lookup_2/Assert/AssertAssert!model/string_lookup_2/Equal_1:z:0+model/string_lookup_2/StringFormat:output:0)^model/movie_index_lookup/Assert_1/Assert*

T
2*
_output_shapes
 ?
model/string_lookup_2/IdentityIdentity<model/string_lookup_2/None_Lookup/LookupTableFindV2:values:0$^model/string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
3model/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handle	age_groupAmodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????h
model/string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
model/string_lookup_1/EqualEqual<model/string_lookup_1/None_Lookup/LookupTableFindV2:values:0&model/string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????n
model/string_lookup_1/WhereWheremodel/string_lookup_1/Equal:z:0*'
_output_shapes
:??????????
model/string_lookup_1/GatherNdGatherNd	age_group#model/string_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
"model/string_lookup_1/StringFormatStringFormat'model/string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.h
model/string_lookup_1/SizeSize#model/string_lookup_1/Where:index:0*
T0	*
_output_shapes
: a
model/string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
model/string_lookup_1/Equal_1Equal#model/string_lookup_1/Size:output:0(model/string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
#model/string_lookup_1/Assert/AssertAssert!model/string_lookup_1/Equal_1:z:0+model/string_lookup_1/StringFormat:output:0$^model/string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
model/string_lookup_1/IdentityIdentity<model/string_lookup_1/None_Lookup/LookupTableFindV2:values:0$^model/string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
1model/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2>model_string_lookup_none_lookup_lookuptablefindv2_table_handlesex?model_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????f
model/string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
model/string_lookup/EqualEqual:model/string_lookup/None_Lookup/LookupTableFindV2:values:0$model/string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????j
model/string_lookup/WhereWheremodel/string_lookup/Equal:z:0*'
_output_shapes
:??????????
model/string_lookup/GatherNdGatherNdsex!model/string_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
 model/string_lookup/StringFormatStringFormat%model/string_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.d
model/string_lookup/SizeSize!model/string_lookup/Where:index:0*
T0	*
_output_shapes
: _
model/string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
model/string_lookup/Equal_1Equal!model/string_lookup/Size:output:0&model/string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
!model/string_lookup/Assert/AssertAssertmodel/string_lookup/Equal_1:z:0)model/string_lookup/StringFormat:output:0$^model/string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
model/string_lookup/IdentityIdentity:model/string_lookup/None_Lookup/LookupTableFindV2:values:0"^model/string_lookup/Assert/Assert*
T0	*'
_output_shapes
:?????????x
model/dropout_1/IdentityIdentitymodel/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????>?
$model/sex_embedding/embedding_lookupResourceGather*model_sex_embedding_embedding_lookup_20959%model/string_lookup/Identity:output:0*
Tindices0	*=
_class3
1/loc:@model/sex_embedding/embedding_lookup/20959*+
_output_shapes
:?????????*
dtype0?
-model/sex_embedding/embedding_lookup/IdentityIdentity-model/sex_embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@model/sex_embedding/embedding_lookup/20959*+
_output_shapes
:??????????
/model/sex_embedding/embedding_lookup/Identity_1Identity6model/sex_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
*model/age_group_embedding/embedding_lookupResourceGather0model_age_group_embedding_embedding_lookup_20964'model/string_lookup_1/Identity:output:0*
Tindices0	*C
_class9
75loc:@model/age_group_embedding/embedding_lookup/20964*+
_output_shapes
:?????????*
dtype0?
3model/age_group_embedding/embedding_lookup/IdentityIdentity3model/age_group_embedding/embedding_lookup:output:0*
T0*C
_class9
75loc:@model/age_group_embedding/embedding_lookup/20964*+
_output_shapes
:??????????
5model/age_group_embedding/embedding_lookup/Identity_1Identity<model/age_group_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
+model/occupation_embedding/embedding_lookupResourceGather1model_occupation_embedding_embedding_lookup_20969'model/string_lookup_2/Identity:output:0*
Tindices0	*D
_class:
86loc:@model/occupation_embedding/embedding_lookup/20969*+
_output_shapes
:?????????*
dtype0?
4model/occupation_embedding/embedding_lookup/IdentityIdentity4model/occupation_embedding/embedding_lookup:output:0*
T0*D
_class:
86loc:@model/occupation_embedding/embedding_lookup/20969*+
_output_shapes
:??????????
6model/occupation_embedding/embedding_lookup/Identity_1Identity=model/occupation_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
model/add_1/addAddV2!model/layer_normalization/add:z:0!model/dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????>_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate/concatConcatV28model/sex_embedding/embedding_lookup/Identity_1:output:0>model/age_group_embedding/embedding_lookup/Identity_1:output:0?model/occupation_embedding/embedding_lookup/Identity_1:output:0&model/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????d
!model/layer_normalization_1/ShapeShapemodel/add_1/add:z:0*
T0*
_output_shapes
:y
/model/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model/layer_normalization_1/strided_sliceStridedSlice*model/layer_normalization_1/Shape:output:08model/layer_normalization_1/strided_slice/stack:output:0:model/layer_normalization_1/strided_slice/stack_1:output:0:model/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
model/layer_normalization_1/mulMul*model/layer_normalization_1/mul/x:output:02model/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model/layer_normalization_1/strided_slice_1StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_1/stack:output:0<model/layer_normalization_1/strided_slice_1/stack_1:output:0<model/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!model/layer_normalization_1/mul_1Mul#model/layer_normalization_1/mul:z:04model/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: {
1model/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model/layer_normalization_1/strided_slice_2StridedSlice*model/layer_normalization_1/Shape:output:0:model/layer_normalization_1/strided_slice_2/stack:output:0<model/layer_normalization_1/strided_slice_2/stack_1:output:0<model/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :?
!model/layer_normalization_1/mul_2Mul,model/layer_normalization_1/mul_2/x:output:04model/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: m
+model/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+model/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)model/layer_normalization_1/Reshape/shapePack4model/layer_normalization_1/Reshape/shape/0:output:0%model/layer_normalization_1/mul_1:z:0%model/layer_normalization_1/mul_2:z:04model/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#model/layer_normalization_1/ReshapeReshapemodel/add_1/add:z:02model/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????>?
'model/layer_normalization_1/ones/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:k
&model/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 model/layer_normalization_1/onesFill0model/layer_normalization_1/ones/packed:output:0/model/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:??????????
(model/layer_normalization_1/zeros/packedPack%model/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:l
'model/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!model/layer_normalization_1/zerosFill1model/layer_normalization_1/zeros/packed:output:00model/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????d
!model/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#model/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
,model/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3,model/layer_normalization_1/Reshape:output:0)model/layer_normalization_1/ones:output:0*model/layer_normalization_1/zeros:output:0*model/layer_normalization_1/Const:output:0,model/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????>:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
%model/layer_normalization_1/Reshape_1Reshape0model/layer_normalization_1/FusedBatchNormV3:y:0*model/layer_normalization_1/Shape:output:0*
T0*+
_output_shapes
:?????????>?
0model/layer_normalization_1/mul_3/ReadVariableOpReadVariableOp9model_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes
:>*
dtype0?
!model/layer_normalization_1/mul_3Mul.model/layer_normalization_1/Reshape_1:output:08model/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>?
.model/layer_normalization_1/add/ReadVariableOpReadVariableOp7model_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:>*
dtype0?
model/layer_normalization_1/addAddV2%model/layer_normalization_1/mul_3:z:06model/layer_normalization_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????D  ?
model/flatten/ReshapeReshape#model/layer_normalization_1/add:z:0model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????d
model/reshape/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
model/reshape/ReshapeReshape!model/concatenate/concat:output:0$model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????a
model/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model/concatenate_4/concatConcatV2model/flatten/Reshape:output:0model/reshape/Reshape:output:0(model/concatenate_4/concat/axis:output:0*
N*
T0*(
_output_shapes
:???????????
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model/dense_1/MatMulMatMul#model/concatenate_4/concat:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:??
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/mul_1Mulmodel/dense_1/BiasAdd:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
model/leaky_re_lu_1/LeakyRelu	LeakyRelu-model/batch_normalization/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>?
model/dropout_2/IdentityIdentity+model/leaky_re_lu_1/LeakyRelu:activations:0*
T0*(
_output_shapes
:???????????
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model/dense_2/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:?
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:??
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/mul_1Mulmodel/dense_2/BiasAdd:output:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:???????????
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:??
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0?
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:??
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:???????????
model/leaky_re_lu_2/LeakyRelu	LeakyRelu/model/batch_normalization_1/batchnorm/add_1:z:0*(
_output_shapes
:??????????*
alpha%???>?
model/dropout_3/IdentityIdentity+model/leaky_re_lu_2/LeakyRelu:activations:0*
T0*(
_output_shapes
:???????????
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model/dense_3/MatMulMatMul!model/dropout_3/Identity:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp+^model/age_group_embedding/embedding_lookup3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp%^model/genres_vector/embedding_lookup'^model/genres_vector/embedding_lookup_1-^model/layer_normalization/add/ReadVariableOp/^model/layer_normalization/mul_3/ReadVariableOp/^model/layer_normalization_1/add/ReadVariableOp1^model/layer_normalization_1/mul_3/ReadVariableOp'^model/movie_embedding/embedding_lookup)^model/movie_embedding/embedding_lookup_1'^model/movie_index_lookup/Assert/Assert)^model/movie_index_lookup/Assert_1/Assert7^model/movie_index_lookup/None_Lookup/LookupTableFindV29^model/movie_index_lookup/None_Lookup_1/LookupTableFindV2?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOp,^model/occupation_embedding/embedding_lookupA^model/process_movie_embedding_with_genres/BiasAdd/ReadVariableOpC^model/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOpC^model/process_movie_embedding_with_genres/Tensordot/ReadVariableOpE^model/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp%^model/sex_embedding/embedding_lookup"^model/string_lookup/Assert/Assert2^model/string_lookup/None_Lookup/LookupTableFindV2$^model/string_lookup_1/Assert/Assert4^model/string_lookup_1/None_Lookup/LookupTableFindV2$^model/string_lookup_2/Assert/Assert4^model/string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*model/age_group_embedding/embedding_lookup*model/age_group_embedding/embedding_lookup2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2L
$model/genres_vector/embedding_lookup$model/genres_vector/embedding_lookup2P
&model/genres_vector/embedding_lookup_1&model/genres_vector/embedding_lookup_12\
,model/layer_normalization/add/ReadVariableOp,model/layer_normalization/add/ReadVariableOp2`
.model/layer_normalization/mul_3/ReadVariableOp.model/layer_normalization/mul_3/ReadVariableOp2`
.model/layer_normalization_1/add/ReadVariableOp.model/layer_normalization_1/add/ReadVariableOp2d
0model/layer_normalization_1/mul_3/ReadVariableOp0model/layer_normalization_1/mul_3/ReadVariableOp2P
&model/movie_embedding/embedding_lookup&model/movie_embedding/embedding_lookup2T
(model/movie_embedding/embedding_lookup_1(model/movie_embedding/embedding_lookup_12P
&model/movie_index_lookup/Assert/Assert&model/movie_index_lookup/Assert/Assert2T
(model/movie_index_lookup/Assert_1/Assert(model/movie_index_lookup/Assert_1/Assert2p
6model/movie_index_lookup/None_Lookup/LookupTableFindV26model/movie_index_lookup/None_Lookup/LookupTableFindV22t
8model/movie_index_lookup/None_Lookup_1/LookupTableFindV28model/movie_index_lookup/None_Lookup_1/LookupTableFindV22?
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2?
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2Z
+model/occupation_embedding/embedding_lookup+model/occupation_embedding/embedding_lookup2?
@model/process_movie_embedding_with_genres/BiasAdd/ReadVariableOp@model/process_movie_embedding_with_genres/BiasAdd/ReadVariableOp2?
Bmodel/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOpBmodel/process_movie_embedding_with_genres/BiasAdd_1/ReadVariableOp2?
Bmodel/process_movie_embedding_with_genres/Tensordot/ReadVariableOpBmodel/process_movie_embedding_with_genres/Tensordot/ReadVariableOp2?
Dmodel/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOpDmodel/process_movie_embedding_with_genres/Tensordot_1/ReadVariableOp2L
$model/sex_embedding/embedding_lookup$model/sex_embedding/embedding_lookup2F
!model/string_lookup/Assert/Assert!model/string_lookup/Assert/Assert2f
1model/string_lookup/None_Lookup/LookupTableFindV21model/string_lookup/None_Lookup/LookupTableFindV22J
#model/string_lookup_1/Assert/Assert#model/string_lookup_1/Assert/Assert2j
3model/string_lookup_1/None_Lookup/LookupTableFindV23model/string_lookup_1/None_Lookup/LookupTableFindV22J
#model/string_lookup_2/Assert/Assert#model/string_lookup_2/Assert/Assert2j
3model/string_lookup_2/None_Lookup/LookupTableFindV23model/string_lookup_2/None_Lookup/LookupTableFindV2:R N
'
_output_shapes
:?????????
#
_user_specified_name	age_group:SO
'
_output_shapes
:?????????
$
_user_specified_name
occupation:[W
'
_output_shapes
:?????????
,
_user_specified_namesequence_movie_ids:YU
'
_output_shapes
:?????????
*
_user_specified_namesequence_ratings:LH
'
_output_shapes
:?????????

_user_specified_namesex:XT
'
_output_shapes
:?????????
)
_user_specified_nametarget_movie_id:PL
'
_output_shapes
:?????????
!
_user_specified_name	user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
j
>__inference_add_layer_call_and_return_conditional_losses_25334
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_21950

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781

inputs	(
embedding_lookup_21775:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21775inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21775*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21775*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25105
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :{
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
4__inference_multi_head_attention_layer_call_fn_25218	
query	
value
unknown:>>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_22402s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:?????????>:?????????>: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:?????????>

_user_specified_namequery:RN
+
_output_shapes
:?????????>

_user_specified_namevalue
?
,
__inference__destroyer_25947
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25049

inputs3
!tensordot_readvariableop_resource:P>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:P>*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????P?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:>Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????>r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????>T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????>e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????>z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?

%__inference_model_layer_call_fn_23829
inputs_age_group
inputs_occupation
inputs_sequence_movie_ids
inputs_sequence_ratings

inputs_sex
inputs_target_movie_id
inputs_user_id
unknown
	unknown_0	
	unknown_1:	?>
	unknown_2:	?
	unknown_3:P>
	unknown_4:>
	unknown_5
	unknown_6:>>
	unknown_7:>
	unknown_8:>>
	unknown_9:> 

unknown_10:>>

unknown_11:> 

unknown_12:>>

unknown_13:>

unknown_14:>

unknown_15:>

unknown_16:>>

unknown_17:>

unknown_18

unknown_19	

unknown_20

unknown_21	

unknown_22

unknown_23	

unknown_24:

unknown_25:

unknown_26:

unknown_27:>

unknown_28:>

unknown_29:
??

unknown_30:	?

unknown_31:	?

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:
??

unknown_36:	?

unknown_37:	?

unknown_38:	?

unknown_39:	?

unknown_40:	?

unknown_41:	?

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_age_groupinputs_occupationinputs_sequence_movie_idsinputs_sequence_ratings
inputs_sexinputs_target_movie_idinputs_user_idunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*>
Tin7
523				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*E
_read_only_resource_inputs'
%#	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_22008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameinputs/age_group:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:b^
'
_output_shapes
:?????????
3
_user_specified_nameinputs/sequence_movie_ids:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/sequence_ratings:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/target_movie_id:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/user_id:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25009

inputs
unknown:P>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376

inputs	)
embedding_lookup_21370:	?>
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21370inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21370*+
_output_shapes
:?????????>*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21370*+
_output_shapes
:?????????>?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????>w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????>Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794

inputs	(
embedding_lookup_21788:
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_21788inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21788*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21788*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_21962

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
O
#__inference_add_layer_call_fn_25328
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_21631d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/1
?
?
4__inference_occupation_embedding_layer_call_fn_25517

inputs	
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_25670

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21111p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24987

inputs	)
embedding_lookup_24981:	?
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_24981inputs*
Tindices0	*)
_class
loc:@embedding_lookup/24981*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/24981*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :y
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????P[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_22901

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6A
=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleB
>movie_index_lookup_none_lookup_lookuptablefindv2_default_value	(
movie_embedding_22654:	?>&
genres_vector_22657:	?;
)process_movie_embedding_with_genres_22661:P>7
)process_movie_embedding_with_genres_22663:> 
tf___operators___add_addv2_y0
multi_head_attention_22777:>>,
multi_head_attention_22779:>0
multi_head_attention_22781:>>,
multi_head_attention_22783:>0
multi_head_attention_22785:>>,
multi_head_attention_22787:>0
multi_head_attention_22789:>>(
multi_head_attention_22791:>'
layer_normalization_22796:>'
layer_normalization_22798:>
dense_22802:>>
dense_22804:>>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	%
sex_embedding_22844:+
age_group_embedding_22847:,
occupation_embedding_22850:)
layer_normalization_1_22855:>)
layer_normalization_1_22857:>!
dense_1_22863:
??
dense_1_22865:	?(
batch_normalization_22868:	?(
batch_normalization_22870:	?(
batch_normalization_22872:	?(
batch_normalization_22874:	?!
dense_2_22879:
??
dense_2_22881:	?*
batch_normalization_1_22884:	?*
batch_normalization_1_22886:	?*
batch_normalization_1_22888:	?*
batch_normalization_1_22890:	? 
dense_3_22895:	?
dense_3_22897:
identity??+age_group_embedding/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!dropout_3/StatefulPartitionedCall?%genres_vector/StatefulPartitionedCall?'genres_vector/StatefulPartitionedCall_1?+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?)movie_embedding/StatefulPartitionedCall_1? movie_index_lookup/Assert/Assert?"movie_index_lookup/Assert_1/Assert?0movie_index_lookup/None_Lookup/LookupTableFindV2?2movie_index_lookup/None_Lookup_1/LookupTableFindV2?,multi_head_attention/StatefulPartitionedCall?,occupation_embedding/StatefulPartitionedCall?;process_movie_embedding_with_genres/StatefulPartitionedCall?=process_movie_embedding_with_genres/StatefulPartitionedCall_1?%sex_embedding/StatefulPartitionedCall?string_lookup/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?string_lookup_1/Assert/Assert?-string_lookup_1/None_Lookup/LookupTableFindV2?string_lookup_2/Assert/Assert?-string_lookup_2/None_Lookup/LookupTableFindV2?
0movie_index_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2>movie_index_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????e
movie_index_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/EqualEqual9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0#movie_index_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????h
movie_index_lookup/WhereWheremovie_index_lookup/Equal:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNdGatherNdinputs_2 movie_index_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
movie_index_lookup/StringFormatStringFormat$movie_index_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.b
movie_index_lookup/SizeSize movie_index_lookup/Where:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_1Equal movie_index_lookup/Size:output:0%movie_index_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
 movie_index_lookup/Assert/AssertAssertmovie_index_lookup/Equal_1:z:0(movie_index_lookup/StringFormat:output:0*

T
2*
_output_shapes
 ?
movie_index_lookup/IdentityIdentity9movie_index_lookup/None_Lookup/LookupTableFindV2:values:0!^movie_index_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0movie_embedding_22654*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21291?
%genres_vector/StatefulPartitionedCallStatefulPartitionedCall$movie_index_lookup/Identity:output:0genres_vector_22657*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21304?
concatenate_2/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0.genres_vector/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_21315?
;process_movie_embedding_with_genres/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0)process_movie_embedding_with_genres_22661)process_movie_embedding_with_genres_22663*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21348?
2movie_index_lookup/None_Lookup_1/LookupTableFindV2LookupTableFindV2=movie_index_lookup_none_lookup_lookuptablefindv2_table_handleinputs_5>movie_index_lookup_none_lookup_lookuptablefindv2_default_value1^movie_index_lookup/None_Lookup/LookupTableFindV2*	
Tin0*

Tout0	*'
_output_shapes
:?????????g
movie_index_lookup/Equal_2/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
movie_index_lookup/Equal_2Equal;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0%movie_index_lookup/Equal_2/y:output:0*
T0	*'
_output_shapes
:?????????l
movie_index_lookup/Where_1Wheremovie_index_lookup/Equal_2:z:0*'
_output_shapes
:??????????
movie_index_lookup/GatherNd_1GatherNdinputs_5"movie_index_lookup/Where_1:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
!movie_index_lookup/StringFormat_1StringFormat&movie_index_lookup/GatherNd_1:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.f
movie_index_lookup/Size_1Size"movie_index_lookup/Where_1:index:0*
T0	*
_output_shapes
: ^
movie_index_lookup/Equal_3/yConst*
_output_shapes
: *
dtype0*
value	B : ?
movie_index_lookup/Equal_3Equal"movie_index_lookup/Size_1:output:0%movie_index_lookup/Equal_3/y:output:0*
T0*
_output_shapes
: ?
"movie_index_lookup/Assert_1/AssertAssertmovie_index_lookup/Equal_3:z:0*movie_index_lookup/StringFormat_1:output:0!^movie_index_lookup/Assert/Assert*

T
2*
_output_shapes
 ?
movie_index_lookup/Identity_1Identity;movie_index_lookup/None_Lookup_1/LookupTableFindV2:values:0#^movie_index_lookup/Assert_1/Assert*
T0	*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2Dprocess_movie_embedding_with_genres/StatefulPartitionedCall:output:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:?????????>h
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.expand_dims/ExpandDims
ExpandDimsinputs_3&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:??????????
)movie_embedding/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0movie_embedding_22654*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_movie_embedding_layer_call_and_return_conditional_losses_21376?
'genres_vector/StatefulPartitionedCall_1StatefulPartitionedCall&movie_index_lookup/Identity_1:output:0genres_vector_22657*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_genres_vector_layer_call_and_return_conditional_losses_21387?
multiply/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_multiply_layer_call_and_return_conditional_losses_21396?
concatenate_1/PartitionedCallPartitionedCall2movie_embedding/StatefulPartitionedCall_1:output:00genres_vector/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_21405?
tf.unstack/unstackUnpack!multiply/PartitionedCall:output:0*
T0*?
_output_shapes?
?:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>:?????????>*

axis*	
numa
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_1/ExpandDims
ExpandDimstf.unstack/unstack:output:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_2/ExpandDims
ExpandDimstf.unstack/unstack:output:1(tf.expand_dims_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_3/ExpandDims
ExpandDimstf.unstack/unstack:output:2(tf.expand_dims_3/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_4/ExpandDims
ExpandDimstf.unstack/unstack:output:3(tf.expand_dims_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_5/ExpandDims
ExpandDimstf.unstack/unstack:output:4(tf.expand_dims_5/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_6/ExpandDims
ExpandDimstf.unstack/unstack:output:5(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_7/ExpandDims
ExpandDimstf.unstack/unstack:output:6(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_8/ExpandDims
ExpandDimstf.unstack/unstack:output:7(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>a
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_9/ExpandDims
ExpandDimstf.unstack/unstack:output:8(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_10/ExpandDims
ExpandDimstf.unstack/unstack:output:9)tf.expand_dims_10/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_11/ExpandDims
ExpandDimstf.unstack/unstack:output:10)tf.expand_dims_11/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_12/ExpandDims
ExpandDimstf.unstack/unstack:output:11)tf.expand_dims_12/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_13/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_13/ExpandDims
ExpandDimstf.unstack/unstack:output:12)tf.expand_dims_13/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_14/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_14/ExpandDims
ExpandDimstf.unstack/unstack:output:13)tf.expand_dims_14/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_15/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_15/ExpandDims
ExpandDimstf.unstack/unstack:output:14)tf.expand_dims_15/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_16/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_16/ExpandDims
ExpandDimstf.unstack/unstack:output:15)tf.expand_dims_16/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_17/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_17/ExpandDims
ExpandDimstf.unstack/unstack:output:16)tf.expand_dims_17/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_18/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_18/ExpandDims
ExpandDimstf.unstack/unstack:output:17)tf.expand_dims_18/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_19/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_19/ExpandDims
ExpandDimstf.unstack/unstack:output:18)tf.expand_dims_19/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_20/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_20/ExpandDims
ExpandDimstf.unstack/unstack:output:19)tf.expand_dims_20/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_21/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_21/ExpandDims
ExpandDimstf.unstack/unstack:output:20)tf.expand_dims_21/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_22/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_22/ExpandDims
ExpandDimstf.unstack/unstack:output:21)tf.expand_dims_22/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_23/ExpandDims
ExpandDimstf.unstack/unstack:output:22)tf.expand_dims_23/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_24/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_24/ExpandDims
ExpandDimstf.unstack/unstack:output:23)tf.expand_dims_24/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_25/ExpandDims
ExpandDimstf.unstack/unstack:output:24)tf.expand_dims_25/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_26/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_26/ExpandDims
ExpandDimstf.unstack/unstack:output:25)tf.expand_dims_26/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_27/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_27/ExpandDims
ExpandDimstf.unstack/unstack:output:26)tf.expand_dims_27/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_28/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_28/ExpandDims
ExpandDimstf.unstack/unstack:output:27)tf.expand_dims_28/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>b
 tf.expand_dims_29/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :?
tf.expand_dims_29/ExpandDims
ExpandDimstf.unstack/unstack:output:28)tf.expand_dims_29/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????>?
=process_movie_embedding_with_genres/StatefulPartitionedCall_1StatefulPartitionedCall&concatenate_1/PartitionedCall:output:0)process_movie_embedding_with_genres_22661)process_movie_embedding_with_genres_22663*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_21524?
concatenate_3/PartitionedCallPartitionedCall$tf.expand_dims_1/ExpandDims:output:0$tf.expand_dims_2/ExpandDims:output:0$tf.expand_dims_3/ExpandDims:output:0$tf.expand_dims_4/ExpandDims:output:0$tf.expand_dims_5/ExpandDims:output:0$tf.expand_dims_6/ExpandDims:output:0$tf.expand_dims_7/ExpandDims:output:0$tf.expand_dims_8/ExpandDims:output:0$tf.expand_dims_9/ExpandDims:output:0%tf.expand_dims_10/ExpandDims:output:0%tf.expand_dims_11/ExpandDims:output:0%tf.expand_dims_12/ExpandDims:output:0%tf.expand_dims_13/ExpandDims:output:0%tf.expand_dims_14/ExpandDims:output:0%tf.expand_dims_15/ExpandDims:output:0%tf.expand_dims_16/ExpandDims:output:0%tf.expand_dims_17/ExpandDims:output:0%tf.expand_dims_18/ExpandDims:output:0%tf.expand_dims_19/ExpandDims:output:0%tf.expand_dims_20/ExpandDims:output:0%tf.expand_dims_21/ExpandDims:output:0%tf.expand_dims_22/ExpandDims:output:0%tf.expand_dims_23/ExpandDims:output:0%tf.expand_dims_24/ExpandDims:output:0%tf.expand_dims_25/ExpandDims:output:0%tf.expand_dims_26/ExpandDims:output:0%tf.expand_dims_27/ExpandDims:output:0%tf.expand_dims_28/ExpandDims:output:0%tf.expand_dims_29/ExpandDims:output:0Fprocess_movie_embedding_with_genres/StatefulPartitionedCall_1:output:0*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_3_layer_call_and_return_conditional_losses_21563?
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0&concatenate_3/PartitionedCall:output:0multi_head_attention_22777multi_head_attention_22779multi_head_attention_22781multi_head_attention_22783multi_head_attention_22785multi_head_attention_22787multi_head_attention_22789multi_head_attention_22791*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_22402?
dropout/StatefulPartitionedCallStatefulPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0#^movie_index_lookup/Assert_1/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_22331?
add/PartitionedCallPartitionedCall&concatenate_3/PartitionedCall:output:0(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_21631?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_22796layer_normalization_22798*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_21680?
leaky_re_lu/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_21691?
dense/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0dense_22802dense_22804*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_21723?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_1;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_2/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_2/EqualEqual6string_lookup_2/None_Lookup/LookupTableFindV2:values:0 string_lookup_2/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_2/WhereWherestring_lookup_2/Equal:z:0*'
_output_shapes
:??????????
string_lookup_2/GatherNdGatherNdinputs_1string_lookup_2/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_2/StringFormatStringFormat!string_lookup_2/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_2/SizeSizestring_lookup_2/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_2/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_2/Equal_1Equalstring_lookup_2/Size:output:0"string_lookup_2/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_2/Assert/AssertAssertstring_lookup_2/Equal_1:z:0%string_lookup_2/StringFormat:output:0 ^dropout/StatefulPartitionedCall*

T
2*
_output_shapes
 ?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0^string_lookup_2/Assert/Assert*
T0	*'
_output_shapes
:??????????
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????b
string_lookup_1/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup_1/EqualEqual6string_lookup_1/None_Lookup/LookupTableFindV2:values:0 string_lookup_1/Equal/y:output:0*
T0	*'
_output_shapes
:?????????b
string_lookup_1/WhereWherestring_lookup_1/Equal:z:0*'
_output_shapes
:??????????
string_lookup_1/GatherNdGatherNdinputsstring_lookup_1/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup_1/StringFormatStringFormat!string_lookup_1/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.\
string_lookup_1/SizeSizestring_lookup_1/Where:index:0*
T0	*
_output_shapes
: [
string_lookup_1/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
string_lookup_1/Equal_1Equalstring_lookup_1/Size:output:0"string_lookup_1/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup_1/Assert/AssertAssertstring_lookup_1/Equal_1:z:0%string_lookup_1/StringFormat:output:0^string_lookup_2/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0^string_lookup_1/Assert/Assert*
T0	*'
_output_shapes
:??????????
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_49string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????`
string_lookup/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
??????????
string_lookup/EqualEqual4string_lookup/None_Lookup/LookupTableFindV2:values:0string_lookup/Equal/y:output:0*
T0	*'
_output_shapes
:?????????^
string_lookup/WhereWherestring_lookup/Equal:z:0*'
_output_shapes
:??????????
string_lookup/GatherNdGatherNdinputs_4string_lookup/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:??????????
string_lookup/StringFormatStringFormatstring_lookup/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*?
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.X
string_lookup/SizeSizestring_lookup/Where:index:0*
T0	*
_output_shapes
: Y
string_lookup/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
string_lookup/Equal_1Equalstring_lookup/Size:output:0 string_lookup/Equal_1/y:output:0*
T0*
_output_shapes
: ?
string_lookup/Assert/AssertAssertstring_lookup/Equal_1:z:0#string_lookup/StringFormat:output:0^string_lookup_1/Assert/Assert*

T
2*
_output_shapes
 ?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0^string_lookup/Assert/Assert*
T0	*'
_output_shapes
:??????????
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0^string_lookup/Assert/Assert*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_22275?
%sex_embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0sex_embedding_22844*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sex_embedding_layer_call_and_return_conditional_losses_21781?
+age_group_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0age_group_embedding_22847*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_21794?
,occupation_embedding/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0occupation_embedding_22850*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_21807?
add_1/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_21817?
concatenate/PartitionedCallPartitionedCall.sex_embedding/StatefulPartitionedCall:output:04age_group_embedding/StatefulPartitionedCall:output:05occupation_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_21827?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_1_22855layer_normalization_1_22857*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876?
flatten/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_21888?
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_21902?
concatenate_4/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0 reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_21911?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0dense_1_22863dense_1_22865*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_21923?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_22868batch_normalization_22870batch_normalization_22872batch_normalization_22874*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21158?
leaky_re_lu_1/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_21943?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_22168?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_22879dense_2_22881*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_21962?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0batch_normalization_1_22884batch_normalization_1_22886batch_normalization_1_22888batch_normalization_1_22890*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_21240?
leaky_re_lu_2/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_21982?
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_22129?
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_3_22895dense_3_22897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_22001w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^age_group_embedding/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall&^genres_vector/StatefulPartitionedCall(^genres_vector/StatefulPartitionedCall_1,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall*^movie_embedding/StatefulPartitionedCall_1!^movie_index_lookup/Assert/Assert#^movie_index_lookup/Assert_1/Assert1^movie_index_lookup/None_Lookup/LookupTableFindV23^movie_index_lookup/None_Lookup_1/LookupTableFindV2-^multi_head_attention/StatefulPartitionedCall-^occupation_embedding/StatefulPartitionedCall<^process_movie_embedding_with_genres/StatefulPartitionedCall>^process_movie_embedding_with_genres/StatefulPartitionedCall_1&^sex_embedding/StatefulPartitionedCall^string_lookup/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2^string_lookup_1/Assert/Assert.^string_lookup_1/None_Lookup/LookupTableFindV2^string_lookup_2/Assert/Assert.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : :>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+age_group_embedding/StatefulPartitionedCall+age_group_embedding/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2N
%genres_vector/StatefulPartitionedCall%genres_vector/StatefulPartitionedCall2R
'genres_vector/StatefulPartitionedCall_1'genres_vector/StatefulPartitionedCall_12Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2V
)movie_embedding/StatefulPartitionedCall_1)movie_embedding/StatefulPartitionedCall_12D
 movie_index_lookup/Assert/Assert movie_index_lookup/Assert/Assert2H
"movie_index_lookup/Assert_1/Assert"movie_index_lookup/Assert_1/Assert2d
0movie_index_lookup/None_Lookup/LookupTableFindV20movie_index_lookup/None_Lookup/LookupTableFindV22h
2movie_index_lookup/None_Lookup_1/LookupTableFindV22movie_index_lookup/None_Lookup_1/LookupTableFindV22\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2\
,occupation_embedding/StatefulPartitionedCall,occupation_embedding/StatefulPartitionedCall2z
;process_movie_embedding_with_genres/StatefulPartitionedCall;process_movie_embedding_with_genres/StatefulPartitionedCall2~
=process_movie_embedding_with_genres/StatefulPartitionedCall_1=process_movie_embedding_with_genres/StatefulPartitionedCall_12N
%sex_embedding/StatefulPartitionedCall%sex_embedding/StatefulPartitionedCall2:
string_lookup/Assert/Assertstring_lookup/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22>
string_lookup_1/Assert/Assertstring_lookup_1/Assert/Assert2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22>
string_lookup_2/Assert/Assertstring_lookup_2/Assert/Assert2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :$ 

_output_shapes

:>:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_25322

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????>C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????>*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????>s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????>m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????>]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25839

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:???????????
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_layer_normalization_1_layer_call_fn_25535

inputs
unknown:>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_21876s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????>`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
,
__inference__destroyer_25965
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
h
>__inference_add_layer_call_and_return_conditional_losses_21631

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:?????????>S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????>:S O
+
_output_shapes
:?????????>
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_25783

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_21962p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
T
(__inference_multiply_layer_call_fn_25086
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????>* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_multiply_layer_call_and_return_conditional_losses_21396d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????>"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????>:?????????:U Q
+
_output_shapes
:?????????>
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
,
__inference__destroyer_25983
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "?L
saver_filename:0StatefulPartitionedCall_5:0StatefulPartitionedCall_68"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	age_group2
serving_default_age_group:0?????????
A

occupation3
serving_default_occupation:0?????????
Q
sequence_movie_ids;
$serving_default_sequence_movie_ids:0?????????
M
sequence_ratings9
"serving_default_sequence_ratings:0?????????
3
sex,
serving_default_sex:0?????????
K
target_movie_id8
!serving_default_target_movie_id:0?????????
;
user_id0
serving_default_user_id:0?????????=
dense_32
StatefulPartitionedCall_4:0?????????tensorflow/serving/predict:Ѣ
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer_with_weights-3
,layer-43
-layer-44
.layer-45
/layer_with_weights-4
/layer-46
0layer-47
1layer_with_weights-5
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer_with_weights-6
:layer-57
;layer_with_weights-7
;layer-58
<layer_with_weights-8
<layer-59
=layer_with_weights-9
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer_with_weights-10
Blayer-65
Clayer_with_weights-11
Clayer-66
Dlayer-67
Elayer-68
Flayer_with_weights-12
Flayer-69
Glayer_with_weights-13
Glayer-70
Hlayer-71
Ilayer-72
Jlayer-73
Klayer_with_weights-14
Klayer-74
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
R_default_save_signature
S	optimizer
T
signatures"
_tf_keras_network
"
_tf_keras_input_layer
P
U	keras_api
Vinput_vocabulary
Wlookup_table"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^
embeddings"
_tf_keras_layer
?
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e
embeddings"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
?
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
(
t	keras_api"
_tf_keras_layer
(
u	keras_api"
_tf_keras_layer
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
?
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
S
?	keras_api
?input_vocabulary
?lookup_table"
_tf_keras_layer
S
?	keras_api
?input_vocabulary
?lookup_table"
_tf_keras_layer
S
?	keras_api
?input_vocabulary
?lookup_table"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
"
_tf_keras_input_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
^0
e1
r2
s3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34"
trackable_list_wrapper
?
^0
r1
s2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
R_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
%__inference_model_layer_call_fn_22099
%__inference_model_layer_call_fn_23829
%__inference_model_layer_call_fn_23928
%__inference_model_layer_call_fn_23091?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
@__inference_model_layer_call_and_return_conditional_losses_24394
@__inference_model_layer_call_and_return_conditional_losses_24923
@__inference_model_layer_call_and_return_conditional_losses_23359
@__inference_model_layer_call_and_return_conditional_losses_23627?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
 __inference__wrapped_model_21087	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter

?decay
?learning_rate^accumulator?raccumulator?saccumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator??accumulator?"
	optimizer
-
?serving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
'
^0"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
/__inference_movie_embedding_layer_call_fn_24930
/__inference_movie_embedding_layer_call_fn_24937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24946
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
-:+	?>2movie_embedding/embeddings
'
e0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
-__inference_genres_vector_layer_call_fn_24962
-__inference_genres_vector_layer_call_fn_24969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24978
H__inference_genres_vector_layer_call_and_return_conditional_losses_24987?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
+:)	?2genres_vector/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_2_layer_call_fn_24993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_25000?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25009
C__inference_process_movie_embedding_with_genres_layer_call_fn_25018?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25049
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25080?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
<::P>2*process_movie_embedding_with_genres/kernel
6:4>2(process_movie_embedding_with_genres/bias
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_multiply_layer_call_fn_25086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_multiply_layer_call_and_return_conditional_losses_25092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_1_layer_call_fn_25098?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_3_layer_call_fn_25139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_25174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
4__inference_multi_head_attention_layer_call_fn_25196
4__inference_multi_head_attention_layer_call_fn_25218?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25253
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25295?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
'__inference_dropout_layer_call_fn_25300
'__inference_dropout_layer_call_fn_25305?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
B__inference_dropout_layer_call_and_return_conditional_losses_25310
B__inference_dropout_layer_call_and_return_conditional_losses_25322?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
#__inference_add_layer_call_fn_25328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
>__inference_add_layer_call_and_return_conditional_losses_25334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_layer_normalization_layer_call_fn_25343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_25390?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
':%>2layer_normalization/gamma
&:$>2layer_normalization/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_leaky_re_lu_layer_call_fn_25395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_25400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_dense_layer_call_fn_25409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_dense_layer_call_and_return_conditional_losses_25439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
:>>2dense/kernel
:>2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_1_layer_call_fn_25444
)__inference_dropout_1_layer_call_fn_25449?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_1_layer_call_and_return_conditional_losses_25454
D__inference_dropout_1_layer_call_and_return_conditional_losses_25466?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
%__inference_add_1_layer_call_fn_25472?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
@__inference_add_1_layer_call_and_return_conditional_losses_25478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_sex_embedding_layer_call_fn_25485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_sex_embedding_layer_call_and_return_conditional_losses_25494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(2sex_embedding/embeddings
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_age_group_embedding_layer_call_fn_25501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_25510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0:.2age_group_embedding/embeddings
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
4__inference_occupation_embedding_layer_call_fn_25517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_25526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
1:/2occupation_embedding/embeddings
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
5__inference_layer_normalization_1_layer_call_fn_25535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
):'>2layer_normalization_1/gamma
(:&>2layer_normalization_1/beta
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_concatenate_layer_call_fn_25589?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_concatenate_layer_call_and_return_conditional_losses_25597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_flatten_layer_call_fn_25602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_flatten_layer_call_and_return_conditional_losses_25608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_reshape_layer_call_fn_25613?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_reshape_layer_call_and_return_conditional_losses_25625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_4_layer_call_fn_25631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_4_layer_call_and_return_conditional_losses_25638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_1_layer_call_fn_25647?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_1_layer_call_and_return_conditional_losses_25657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
": 
??2dense_1/kernel
:?2dense_1/bias
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
3__inference_batch_normalization_layer_call_fn_25670
3__inference_batch_normalization_layer_call_fn_25683?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25703
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25737?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_leaky_re_lu_1_layer_call_fn_25742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_25747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_2_layer_call_fn_25752
)__inference_dropout_2_layer_call_fn_25757?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25762
D__inference_dropout_2_layer_call_and_return_conditional_losses_25774?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_2_layer_call_fn_25783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_2_layer_call_and_return_conditional_losses_25793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
": 
??2dense_2/kernel
:?2dense_2/bias
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
5__inference_batch_normalization_1_layer_call_fn_25806
5__inference_batch_normalization_1_layer_call_fn_25819?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25839
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25873?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_leaky_re_lu_2_layer_call_fn_25878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_25883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
)__inference_dropout_3_layer_call_fn_25888
)__inference_dropout_3_layer_call_fn_25893?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
D__inference_dropout_3_layer_call_and_return_conditional_losses_25898
D__inference_dropout_3_layer_call_and_return_conditional_losses_25910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_dense_3_layer_call_fn_25919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_dense_3_layer_call_and_return_conditional_losses_25929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:	?2dense_3/kernel
:2dense_3/bias
7:5>>2!multi_head_attention/query/kernel
1:/>2multi_head_attention/query/bias
5:3>>2multi_head_attention/key/kernel
/:->2multi_head_attention/key/bias
7:5>>2!multi_head_attention/value/kernel
1:/>2multi_head_attention/value/bias
B:@>>2,multi_head_attention/attention_output/kernel
8:6>2*multi_head_attention/attention_output/bias
G
e0
?1
?2
?3
?4"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_model_layer_call_fn_22099	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_23829inputs/age_groupinputs/occupationinputs/sequence_movie_idsinputs/sequence_ratings
inputs/sexinputs/target_movie_idinputs/user_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_23928inputs/age_groupinputs/occupationinputs/sequence_movie_idsinputs/sequence_ratings
inputs/sexinputs/target_movie_idinputs/user_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_model_layer_call_fn_23091	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_24394inputs/age_groupinputs/occupationinputs/sequence_movie_idsinputs/sequence_ratings
inputs/sexinputs/target_movie_idinputs/user_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_24923inputs/age_groupinputs/occupationinputs/sequence_movie_idsinputs/sequence_ratings
inputs/sexinputs/target_movie_idinputs/user_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_23359	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
@__inference_model_layer_call_and_return_conditional_losses_23627	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
?B?
#__inference_signature_wrapper_23730	age_group
occupationsequence_movie_idssequence_ratingssextarget_movie_iduser_id"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_25934?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_25942?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_25947?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
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
?B?
/__inference_movie_embedding_layer_call_fn_24930inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
/__inference_movie_embedding_layer_call_fn_24937inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24946inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24955inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
e0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_genres_vector_layer_call_fn_24962inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
-__inference_genres_vector_layer_call_fn_24969inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24978inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24987inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_2_layer_call_fn_24993inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_25000inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25009inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25018inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25049inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25080inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
(__inference_multiply_layer_call_fn_25086inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_multiply_layer_call_and_return_conditional_losses_25092inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_1_layer_call_fn_25098inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25105inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_3_layer_call_fn_25139inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10	inputs/11	inputs/12	inputs/13	inputs/14	inputs/15	inputs/16	inputs/17	inputs/18	inputs/19	inputs/20	inputs/21	inputs/22	inputs/23	inputs/24	inputs/25	inputs/26	inputs/27	inputs/28	inputs/29"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_3_layer_call_and_return_conditional_losses_25174inputs/0inputs/1inputs/2inputs/3inputs/4inputs/5inputs/6inputs/7inputs/8inputs/9	inputs/10	inputs/11	inputs/12	inputs/13	inputs/14	inputs/15	inputs/16	inputs/17	inputs/18	inputs/19	inputs/20	inputs/21	inputs/22	inputs/23	inputs/24	inputs/25	inputs/26	inputs/27	inputs/28	inputs/29"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_multi_head_attention_layer_call_fn_25196queryvalue"?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
4__inference_multi_head_attention_layer_call_fn_25218queryvalue"?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25253queryvalue"?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25295queryvalue"?
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_dropout_layer_call_fn_25300inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_dropout_layer_call_fn_25305inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_25310inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_dropout_layer_call_and_return_conditional_losses_25322inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
#__inference_add_layer_call_fn_25328inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
>__inference_add_layer_call_and_return_conditional_losses_25334inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
3__inference_layer_normalization_layer_call_fn_25343inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_25390inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
+__inference_leaky_re_lu_layer_call_fn_25395inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_25400inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
%__inference_dense_layer_call_fn_25409inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_dense_layer_call_and_return_conditional_losses_25439inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_dropout_1_layer_call_fn_25444inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_1_layer_call_fn_25449inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_25454inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_1_layer_call_and_return_conditional_losses_25466inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
?
?trace_02?
__inference__creator_25952?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_25960?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_25965?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_25970?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_25978?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_25983?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
"
_generic_user_object
?
?trace_02?
__inference__creator_25988?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_25996?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_26001?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
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
?B?
%__inference_add_1_layer_call_fn_25472inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
@__inference_add_1_layer_call_and_return_conditional_losses_25478inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_sex_embedding_layer_call_fn_25485inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_sex_embedding_layer_call_and_return_conditional_losses_25494inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
3__inference_age_group_embedding_layer_call_fn_25501inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_25510inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
4__inference_occupation_embedding_layer_call_fn_25517inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_25526inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
5__inference_layer_normalization_1_layer_call_fn_25535inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25582inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
+__inference_concatenate_layer_call_fn_25589inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_concatenate_layer_call_and_return_conditional_losses_25597inputs/0inputs/1inputs/2"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_flatten_layer_call_fn_25602inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_flatten_layer_call_and_return_conditional_losses_25608inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_reshape_layer_call_fn_25613inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_reshape_layer_call_and_return_conditional_losses_25625inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_4_layer_call_fn_25631inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_4_layer_call_and_return_conditional_losses_25638inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
'__inference_dense_1_layer_call_fn_25647inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_1_layer_call_and_return_conditional_losses_25657inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
3__inference_batch_normalization_layer_call_fn_25670inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
3__inference_batch_normalization_layer_call_fn_25683inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25703inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25737inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_leaky_re_lu_1_layer_call_fn_25742inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_25747inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_dropout_2_layer_call_fn_25752inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_2_layer_call_fn_25757inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25762inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25774inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
'__inference_dense_2_layer_call_fn_25783inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_2_layer_call_and_return_conditional_losses_25793inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
5__inference_batch_normalization_1_layer_call_fn_25806inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
5__inference_batch_normalization_1_layer_call_fn_25819inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25839inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25873inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_leaky_re_lu_2_layer_call_fn_25878inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_25883inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_dropout_3_layer_call_fn_25888inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
)__inference_dropout_3_layer_call_fn_25893inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_3_layer_call_and_return_conditional_losses_25898inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
D__inference_dropout_3_layer_call_and_return_conditional_losses_25910inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
'__inference_dense_3_layer_call_fn_25919inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_dense_3_layer_call_and_return_conditional_losses_25929inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
?B?
__inference__creator_25934"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_25942"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_25947"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
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
?B?
__inference__creator_25952"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_25960"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_25965"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_25970"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_25978"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_25983"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_25988"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_25996"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_26001"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?:=	?>2.Adagrad/movie_embedding/embeddings/accumulator
N:LP>2>Adagrad/process_movie_embedding_with_genres/kernel/accumulator
H:F>2<Adagrad/process_movie_embedding_with_genres/bias/accumulator
9:7>2-Adagrad/layer_normalization/gamma/accumulator
8:6>2,Adagrad/layer_normalization/beta/accumulator
0:.>>2 Adagrad/dense/kernel/accumulator
*:(>2Adagrad/dense/bias/accumulator
<::2,Adagrad/sex_embedding/embeddings/accumulator
B:@22Adagrad/age_group_embedding/embeddings/accumulator
C:A23Adagrad/occupation_embedding/embeddings/accumulator
;:9>2/Adagrad/layer_normalization_1/gamma/accumulator
::8>2.Adagrad/layer_normalization_1/beta/accumulator
4:2
??2"Adagrad/dense_1/kernel/accumulator
-:+?2 Adagrad/dense_1/bias/accumulator
::8?2-Adagrad/batch_normalization/gamma/accumulator
9:7?2,Adagrad/batch_normalization/beta/accumulator
4:2
??2"Adagrad/dense_2/kernel/accumulator
-:+?2 Adagrad/dense_2/bias/accumulator
<::?2/Adagrad/batch_normalization_1/gamma/accumulator
;:9?2.Adagrad/batch_normalization_1/beta/accumulator
3:1	?2"Adagrad/dense_3/kernel/accumulator
,:*2 Adagrad/dense_3/bias/accumulator
I:G>>25Adagrad/multi_head_attention/query/kernel/accumulator
C:A>23Adagrad/multi_head_attention/query/bias/accumulator
G:E>>23Adagrad/multi_head_attention/key/kernel/accumulator
A:?>21Adagrad/multi_head_attention/key/bias/accumulator
I:G>>25Adagrad/multi_head_attention/value/kernel/accumulator
C:A>23Adagrad/multi_head_attention/value/bias/accumulator
T:R>>2@Adagrad/multi_head_attention/attention_output/kernel/accumulator
J:H>2>Adagrad/multi_head_attention/attention_output/bias/accumulator
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant6
__inference__creator_25934?

? 
? "? 6
__inference__creator_25952?

? 
? "? 6
__inference__creator_25970?

? 
? "? 6
__inference__creator_25988?

? 
? "? 8
__inference__destroyer_25947?

? 
? "? 8
__inference__destroyer_25965?

? 
? "? 8
__inference__destroyer_25983?

? 
? "? 8
__inference__destroyer_26001?

? 
? "? A
__inference__initializer_25942W???

? 
? "? B
__inference__initializer_25960 ????

? 
? "? B
__inference__initializer_25978 ????

? 
? "? B
__inference__initializer_25996 ????

? 
? "? ?
 __inference__wrapped_model_21087?SW?^ers?????????????????????????????????????????
???
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????
? "1?.
,
dense_3!?
dense_3??????????
@__inference_add_1_layer_call_and_return_conditional_losses_25478?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????>
? ")?&
?
0?????????>
? ?
%__inference_add_1_layer_call_fn_25472?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????>
? "??????????>?
>__inference_add_layer_call_and_return_conditional_losses_25334?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????>
? ")?&
?
0?????????>
? ?
#__inference_add_layer_call_fn_25328?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????>
? "??????????>?
N__inference_age_group_embedding_layer_call_and_return_conditional_losses_25510`?/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
3__inference_age_group_embedding_layer_call_fn_25501S?/?,
%?"
 ?
inputs?????????	
? "???????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25839h????4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25873h????4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
5__inference_batch_normalization_1_layer_call_fn_25806[????4?1
*?'
!?
inputs??????????
p 
? "????????????
5__inference_batch_normalization_1_layer_call_fn_25819[????4?1
*?'
!?
inputs??????????
p
? "????????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25703h????4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25737h????4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
3__inference_batch_normalization_layer_call_fn_25670[????4?1
*?'
!?
inputs??????????
p 
? "????????????
3__inference_batch_normalization_layer_call_fn_25683[????4?1
*?'
!?
inputs??????????
p
? "????????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25105?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? ")?&
?
0?????????P
? ?
-__inference_concatenate_1_layer_call_fn_25098?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? "??????????P?
H__inference_concatenate_2_layer_call_and_return_conditional_losses_25000?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? ")?&
?
0?????????P
? ?
-__inference_concatenate_2_layer_call_fn_24993?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? "??????????P?

H__inference_concatenate_3_layer_call_and_return_conditional_losses_25174?
?	??	
?	??	
?	??	
&?#
inputs/0?????????>
&?#
inputs/1?????????>
&?#
inputs/2?????????>
&?#
inputs/3?????????>
&?#
inputs/4?????????>
&?#
inputs/5?????????>
&?#
inputs/6?????????>
&?#
inputs/7?????????>
&?#
inputs/8?????????>
&?#
inputs/9?????????>
'?$
	inputs/10?????????>
'?$
	inputs/11?????????>
'?$
	inputs/12?????????>
'?$
	inputs/13?????????>
'?$
	inputs/14?????????>
'?$
	inputs/15?????????>
'?$
	inputs/16?????????>
'?$
	inputs/17?????????>
'?$
	inputs/18?????????>
'?$
	inputs/19?????????>
'?$
	inputs/20?????????>
'?$
	inputs/21?????????>
'?$
	inputs/22?????????>
'?$
	inputs/23?????????>
'?$
	inputs/24?????????>
'?$
	inputs/25?????????>
'?$
	inputs/26?????????>
'?$
	inputs/27?????????>
'?$
	inputs/28?????????>
'?$
	inputs/29?????????>
? ")?&
?
0?????????>
? ?

-__inference_concatenate_3_layer_call_fn_25139?	?	??	
?	??	
?	??	
&?#
inputs/0?????????>
&?#
inputs/1?????????>
&?#
inputs/2?????????>
&?#
inputs/3?????????>
&?#
inputs/4?????????>
&?#
inputs/5?????????>
&?#
inputs/6?????????>
&?#
inputs/7?????????>
&?#
inputs/8?????????>
&?#
inputs/9?????????>
'?$
	inputs/10?????????>
'?$
	inputs/11?????????>
'?$
	inputs/12?????????>
'?$
	inputs/13?????????>
'?$
	inputs/14?????????>
'?$
	inputs/15?????????>
'?$
	inputs/16?????????>
'?$
	inputs/17?????????>
'?$
	inputs/18?????????>
'?$
	inputs/19?????????>
'?$
	inputs/20?????????>
'?$
	inputs/21?????????>
'?$
	inputs/22?????????>
'?$
	inputs/23?????????>
'?$
	inputs/24?????????>
'?$
	inputs/25?????????>
'?$
	inputs/26?????????>
'?$
	inputs/27?????????>
'?$
	inputs/28?????????>
'?$
	inputs/29?????????>
? "??????????>?
H__inference_concatenate_4_layer_call_and_return_conditional_losses_25638?[?X
Q?N
L?I
#? 
inputs/0??????????
"?
inputs/1?????????
? "&?#
?
0??????????
? ?
-__inference_concatenate_4_layer_call_fn_25631x[?X
Q?N
L?I
#? 
inputs/0??????????
"?
inputs/1?????????
? "????????????
F__inference_concatenate_layer_call_and_return_conditional_losses_25597????
??}
{?x
&?#
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
? ")?&
?
0?????????
? ?
+__inference_concatenate_layer_call_fn_25589????
??}
{?x
&?#
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
? "???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_25657`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
'__inference_dense_1_layer_call_fn_25647S??0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_2_layer_call_and_return_conditional_losses_25793`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
'__inference_dense_2_layer_call_fn_25783S??0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_3_layer_call_and_return_conditional_losses_25929_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
'__inference_dense_3_layer_call_fn_25919R??0?-
&?#
!?
inputs??????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_25439f??3?0
)?&
$?!
inputs?????????>
? ")?&
?
0?????????>
? ?
%__inference_dense_layer_call_fn_25409Y??3?0
)?&
$?!
inputs?????????>
? "??????????>?
D__inference_dropout_1_layer_call_and_return_conditional_losses_25454d7?4
-?*
$?!
inputs?????????>
p 
? ")?&
?
0?????????>
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_25466d7?4
-?*
$?!
inputs?????????>
p
? ")?&
?
0?????????>
? ?
)__inference_dropout_1_layer_call_fn_25444W7?4
-?*
$?!
inputs?????????>
p 
? "??????????>?
)__inference_dropout_1_layer_call_fn_25449W7?4
-?*
$?!
inputs?????????>
p
? "??????????>?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25762^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25774^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_2_layer_call_fn_25752Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_2_layer_call_fn_25757Q4?1
*?'
!?
inputs??????????
p
? "????????????
D__inference_dropout_3_layer_call_and_return_conditional_losses_25898^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_25910^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ~
)__inference_dropout_3_layer_call_fn_25888Q4?1
*?'
!?
inputs??????????
p 
? "???????????~
)__inference_dropout_3_layer_call_fn_25893Q4?1
*?'
!?
inputs??????????
p
? "????????????
B__inference_dropout_layer_call_and_return_conditional_losses_25310d7?4
-?*
$?!
inputs?????????>
p 
? ")?&
?
0?????????>
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_25322d7?4
-?*
$?!
inputs?????????>
p
? ")?&
?
0?????????>
? ?
'__inference_dropout_layer_call_fn_25300W7?4
-?*
$?!
inputs?????????>
p 
? "??????????>?
'__inference_dropout_layer_call_fn_25305W7?4
-?*
$?!
inputs?????????>
p
? "??????????>?
B__inference_flatten_layer_call_and_return_conditional_losses_25608]3?0
)?&
$?!
inputs?????????>
? "&?#
?
0??????????
? {
'__inference_flatten_layer_call_fn_25602P3?0
)?&
$?!
inputs?????????>
? "????????????
H__inference_genres_vector_layer_call_and_return_conditional_losses_24978_e/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
H__inference_genres_vector_layer_call_and_return_conditional_losses_24987_e/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
-__inference_genres_vector_layer_call_fn_24962Re/?,
%?"
 ?
inputs?????????	
? "???????????
-__inference_genres_vector_layer_call_fn_24969Re/?,
%?"
 ?
inputs?????????	
? "???????????
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25582f??3?0
)?&
$?!
inputs?????????>
? ")?&
?
0?????????>
? ?
5__inference_layer_normalization_1_layer_call_fn_25535Y??3?0
)?&
$?!
inputs?????????>
? "??????????>?
N__inference_layer_normalization_layer_call_and_return_conditional_losses_25390f??3?0
)?&
$?!
inputs?????????>
? ")?&
?
0?????????>
? ?
3__inference_layer_normalization_layer_call_fn_25343Y??3?0
)?&
$?!
inputs?????????>
? "??????????>?
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_25747Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
-__inference_leaky_re_lu_1_layer_call_fn_25742M0?-
&?#
!?
inputs??????????
? "????????????
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_25883Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
-__inference_leaky_re_lu_2_layer_call_fn_25878M0?-
&?#
!?
inputs??????????
? "????????????
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_25400`3?0
)?&
$?!
inputs?????????>
? ")?&
?
0?????????>
? ?
+__inference_leaky_re_lu_layer_call_fn_25395S3?0
)?&
$?!
inputs?????????>
? "??????????>?
@__inference_model_layer_call_and_return_conditional_losses_23359?SW?^ers?????????????????????????????????????????
???
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_23627?SW?^ers?????????????????????????????????????????
???
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_24394?SW?^ers?????????????????????????????????????????
???
???
7
	age_group*?'
inputs/age_group?????????
9

occupation+?(
inputs/occupation?????????
I
sequence_movie_ids3?0
inputs/sequence_movie_ids?????????
E
sequence_ratings1?.
inputs/sequence_ratings?????????
+
sex$?!

inputs/sex?????????
C
target_movie_id0?-
inputs/target_movie_id?????????
3
user_id(?%
inputs/user_id?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_24923?SW?^ers?????????????????????????????????????????
???
???
7
	age_group*?'
inputs/age_group?????????
9

occupation+?(
inputs/occupation?????????
I
sequence_movie_ids3?0
inputs/sequence_movie_ids?????????
E
sequence_ratings1?.
inputs/sequence_ratings?????????
+
sex$?!

inputs/sex?????????
C
target_movie_id0?-
inputs/target_movie_id?????????
3
user_id(?%
inputs/user_id?????????
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_22099?SW?^ers?????????????????????????????????????????
???
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_23091?SW?^ers?????????????????????????????????????????
???
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????
p

 
? "???????????
%__inference_model_layer_call_fn_23829?SW?^ers?????????????????????????????????????????
???
???
7
	age_group*?'
inputs/age_group?????????
9

occupation+?(
inputs/occupation?????????
I
sequence_movie_ids3?0
inputs/sequence_movie_ids?????????
E
sequence_ratings1?.
inputs/sequence_ratings?????????
+
sex$?!

inputs/sex?????????
C
target_movie_id0?-
inputs/target_movie_id?????????
3
user_id(?%
inputs/user_id?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_23928?SW?^ers?????????????????????????????????????????
???
???
7
	age_group*?'
inputs/age_group?????????
9

occupation+?(
inputs/occupation?????????
I
sequence_movie_ids3?0
inputs/sequence_movie_ids?????????
E
sequence_ratings1?.
inputs/sequence_ratings?????????
+
sex$?!

inputs/sex?????????
C
target_movie_id0?-
inputs/target_movie_id?????????
3
user_id(?%
inputs/user_id?????????
p

 
? "???????????
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24946_^/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????>
? ?
J__inference_movie_embedding_layer_call_and_return_conditional_losses_24955_^/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????>
? ?
/__inference_movie_embedding_layer_call_fn_24930R^/?,
%?"
 ?
inputs?????????	
? "??????????>?
/__inference_movie_embedding_layer_call_fn_24937R^/?,
%?"
 ?
inputs?????????	
? "??????????>?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25253?????????g?d
]?Z
#? 
query?????????>
#? 
value?????????>

 

 
p 
p 
? ")?&
?
0?????????>
? ?
O__inference_multi_head_attention_layer_call_and_return_conditional_losses_25295?????????g?d
]?Z
#? 
query?????????>
#? 
value?????????>

 

 
p 
p
? ")?&
?
0?????????>
? ?
4__inference_multi_head_attention_layer_call_fn_25196?????????g?d
]?Z
#? 
query?????????>
#? 
value?????????>

 

 
p 
p 
? "??????????>?
4__inference_multi_head_attention_layer_call_fn_25218?????????g?d
]?Z
#? 
query?????????>
#? 
value?????????>

 

 
p 
p
? "??????????>?
C__inference_multiply_layer_call_and_return_conditional_losses_25092?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? ")?&
?
0?????????>
? ?
(__inference_multiply_layer_call_fn_25086?b?_
X?U
S?P
&?#
inputs/0?????????>
&?#
inputs/1?????????
? "??????????>?
O__inference_occupation_embedding_layer_call_and_return_conditional_losses_25526`?/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
4__inference_occupation_embedding_layer_call_fn_25517S?/?,
%?"
 ?
inputs?????????	
? "???????????
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25049drs3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????>
? ?
^__inference_process_movie_embedding_with_genres_layer_call_and_return_conditional_losses_25080drs3?0
)?&
$?!
inputs?????????P
? ")?&
?
0?????????>
? ?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25009Wrs3?0
)?&
$?!
inputs?????????P
? "??????????>?
C__inference_process_movie_embedding_with_genres_layer_call_fn_25018Wrs3?0
)?&
$?!
inputs?????????P
? "??????????>?
B__inference_reshape_layer_call_and_return_conditional_losses_25625\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? z
'__inference_reshape_layer_call_fn_25613O3?0
)?&
$?!
inputs?????????
? "???????????
H__inference_sex_embedding_layer_call_and_return_conditional_losses_25494`?/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
-__inference_sex_embedding_layer_call_fn_25485S?/?,
%?"
 ?
inputs?????????	
? "???????????
#__inference_signature_wrapper_23730?SW?^ers?????????????????????????????????????????
? 
???
0
	age_group#? 
	age_group?????????
2

occupation$?!

occupation?????????
B
sequence_movie_ids,?)
sequence_movie_ids?????????
>
sequence_ratings*?'
sequence_ratings?????????
$
sex?
sex?????????
<
target_movie_id)?&
target_movie_id?????????
,
user_id!?
user_id?????????"1?.
,
dense_3!?
dense_3?????????