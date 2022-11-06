import pytest
import os, sys
import xfmreadout.obj as obj

from xfmreadout.parser import readfileheader

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
sys.path.insert(0, )

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

USER_CONFIG='config.yaml'
PACKAGE_CONFIG='../xfmreadout/protocol.yaml'

@pytest.fixture
def xfmap():
    return obj.Xfmap(config, fi, fsub)




"""
tests:
    pull single pixel from subts2
        -> save in separate file

    read that pixel as stream
    T pull header and compare to correct vals
    T pull spectrum and compare to reference

    integration:
    
    parse subts2
    -> compare RGB and sum spectrum 
    write 10x10 from subts2
    parse subsubts2
    -> compare RGB and sum spectrum



"""