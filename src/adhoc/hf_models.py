# Code adapted from SimCSE (https://github.com/princeton-nlp/SimCSE) governed by MIT license.

import transformers

model = transformers.AutoModel.from_pretrained('sshleifer/tiny-gpt2')
print(model.num_parameters())
model = transformers.AutoModel.from_pretrained('taufeeque/tiny-gpt2')
print(model.num_parameters())
model = transformers.AutoModel.from_pretrained('google/t5-small-lm-adapt')
print(model.num_parameters())
