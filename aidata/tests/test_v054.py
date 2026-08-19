import os
import tempfile
import numpy as np
import pytest
from aidata import AIDATAWriter, AIDATAReader
from aidata.exceptions import AIDATAError

def test_zlib_roundtrip_without_external_dependency():
    with tempfile.TemporaryDirectory() as tmp:
        path=os.path.join(tmp,"x.aidata")
        X=np.arange(4000,dtype=np.float32).reshape(1000,4)
        y=np.arange(1000,dtype=np.int64)
        AIDATAWriter(path).write(X,y,chunk_size=128,compression=True,compression_level=6,verbose=False)
        with AIDATAReader(path) as r:
            assert r.metadata["compression"]=="zlib"
            assert r.metadata["compression_level"]==6
            xr,yr=r.get_batch(100,200)
            np.testing.assert_array_equal(xr,X[100:300])
            np.testing.assert_array_equal(yr,y[100:300])

def test_uncompressed_level_zero():
    with tempfile.TemporaryDirectory() as tmp:
        path=os.path.join(tmp,"x.aidata")
        X=np.ones((8,2),dtype=np.float32); y=np.arange(8)
        AIDATAWriter(path).write(X,y,compression=False,compression_level=0,verbose=False)
        with AIDATAReader(path) as r:
            assert r.metadata["compression"]=="none"
            assert r.metadata["compression_level"]==0

def test_invalid_zlib_level():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp,"x.aidata")).write(
                np.ones((2,2),dtype=np.float32), np.ones(2,dtype=np.int64),
                compression=True, compression_level=22, verbose=False)

def test_compression_level_reserved():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AIDATAError):
            AIDATAWriter(os.path.join(tmp,"x.aidata")).write(
                np.ones((2,2),dtype=np.float32), np.ones(2,dtype=np.int64),
                metadata={"compression_level": 9}, verbose=False)

def test_pytorch_batch_integration():
    torch=pytest.importorskip("torch")
    from aidata.integrations.pytorch import AIDATAPyTorchDataset, AIDATABatchDataset
    with tempfile.TemporaryDirectory() as tmp:
        path=os.path.join(tmp,"x.aidata")
        X=np.arange(40,dtype=np.float32).reshape(20,2); y=np.arange(20,dtype=np.int64)
        AIDATAWriter(path).write(X,y,compression=True,verbose=False)
        ds=AIDATAPyTorchDataset(path)
        batches=AIDATABatchDataset(ds,batch_size=6)
        xb,yb=batches[3]
        assert tuple(xb.shape)==(2,2)
        assert tuple(yb.shape)==(2,)
        ds.close()
