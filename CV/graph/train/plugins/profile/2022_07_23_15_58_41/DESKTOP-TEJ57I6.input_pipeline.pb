	??W?2?@??W?2?@!??W?2?@	s?$??@s?$??@!s?$??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??W?2?@z?,C???A|a2U?@YP??n???*	?????ly@2F
Iterator::ModelΈ?????!?i?AJ!V@)	?c?Z??1d˘?`wU@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?(??0??!I?T0@){?G?z??1NQ?9j?@:Preprocessing2U
Iterator::Model::ParallelMapV2??_vO??!??S+=@)??_vO??1??S+=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF%u???!?V?m?	@)?J?4??1?N??? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipEGr????!5?d???&@)S?!?uq{?11???Z??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*?s?!?.?	???)a2U0*?s?1?.?	???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?r?!??vh???)HP?s?r?1??vh???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ?	???!?:???H@)/n??b?1;???HN??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9s?$??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z?,C???z?,C???!z?,C???      ??!       "      ??!       *      ??!       2	|a2U?@|a2U?@!|a2U?@:      ??!       B      ??!       J	P??n???P??n???!P??n???R      ??!       Z	P??n???P??n???!P??n???JCPU_ONLYYs?$??@b 