from src.mteb import MTEB
from src.model.inbatch import InBatch
from src.utils.training_utils import get_inbatch_base_config, reload_model_from_ckpt

base_model_name = 'bert-base-uncased'
pooling = 'average'
model_name = 'cc.RC20+T0gen80.moco-2e16.contriever256.bert-base-uncased.avg.dot.len256qd128.step200k.bs4096.lr5e5'
model_path = f'/export/home/exp/search/unsup_dr/cc_v2/{model_name}'
output_path = f'/export/home/exp/search/mteb/'

model_args = get_inbatch_base_config()
model_args.model_name_or_path = base_model_name
model_args.pooling = pooling
model = InBatch(model_args)
model = reload_model_from_ckpt(model, model_path)
model = model.to('cuda:0')
# evaluation = MTEB(tasks=["Banking77Classification"])
# evaluation = MTEB(task_types=['Clustering', 'Retrieval'])
evaluation = MTEB(task_types=["Classification", "Clustering", "PairClassification", "Reranking", "STS", "Summarization"])
evaluation = MTEB(task_types=["Summarization"])
results = evaluation.run(model, output_folder=f"{output_path}/{model_name}", batch_size=32)

print(results)
