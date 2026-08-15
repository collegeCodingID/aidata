# aidata

mkdir -p aidata/src/aidata/integrations
mkdir -p aidata/examples
mkdir -p aidata/tests
cd aidata

aidata/
├── benchmarks/
│   ├── benchmark.py
│   └── training_benchmark.py
│
├── examples/
│   ├── basic.py
│   ├── train_test.py
│   ├── pytorch_train.py
│   └── pytorch_dataloader.py
│
├── src/
│   └── aidata/
│       ├── __init__.py
│       ├── writer.py
│       ├── reader.py
│       ├── dataset.py
│       ├── loader.py
│       └── integrations/
│           ├── __init__.py
│           └── pytorch.py
│
├── tests/
│   └── test_basic.py
│
├── pyproject.toml
└── README.md


 src/aidata/exceptions.py 
  examples/profile_training.py 

  cd aidata
pip install -e .
pytest tests/test_basic.py -v
