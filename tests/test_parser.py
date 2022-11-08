import pytest
import sys, os

#https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
#https://stackoverflow.com/questions/25827160/importing-correctly-with-pytest
TEST_DIR=os.path.realpath(os.path.dirname(__file__))
BASE_DIR=os.path.dirname(TEST_DIR)
sys.path.append(BASE_DIR)

import xfmreadout.obj as obj
import xfmreadout.utils as utils
import xfmreadout.parser as parser

USER_CONFIG=os.path.join(TEST_DIR,'parser_config.yaml')
PACKAGE_CONFIG=os.path.join(TEST_DIR,'parser_protocol.yaml')


#get command line arguments
args = utils.readargs()

#create input config from args and config files
config, rawconfig=utils.initcfg(args, PACKAGE_CONFIG, USER_CONFIG)

#initialise read file and all directories relative to current script
config, fi, fname, fsub, odir = utils.initf(config)

@pytest.fixture
def xf2map():       
    return obj.Xfmap(config, fi, fsub)

#debug hack - not sure how to debug as test, turn it into conventional object rather than fixture for now
#remove "2" from xf2map above
xfmap=obj.Xfmap(config, fi, fsub)


def test_getstream(xfmap):
    """
    not working yet
    """
    headstream, xfmap.idx = parser.getstream(xfmap,xfmap.idx,xfmap.PXHEADERLEN)


def test_read_pxheader(xfmap):
    """
    tests pixel header read
    """
    expected = [5376, 0, 0, 0, 0.0] #known result from first pixel in test datafile

    headstream, xfmap.idx = parser.getstream(xfmap,xfmap.idx,xfmap.PXHEADERLEN)
    pxlen, xidx, yidx, det, dt = parser.readpxheader(headstream, config, xfmap.PXHEADERLEN, xfmap)
    result = [pxlen, xidx, yidx, det, dt]

    assert result == expected


def test_read_pxdata(xfmap):
     """
    not working yet
    """
    #exp_chan = readcsv
    #exp_counts = readcsv
    #pxstart=NUM
    readlength=5376-xfmap.PXHEADERLEN

    xfmap.idx = pxstart
    locstream, xfmap.idx = parser.getstream(xfmap,xfmap.idx,readlength)
    chan, counts = parser.readpxdata(locstream, config, readlength)
    
    assert chan, counts == exp_chan, exp_counts

def test_gapfill():
    """
    not working yet
    """
    #in_chan=readcsv
    #in_chan=readcsv
    #exp_chan=readcsv
    #exp_counts=readcsv

    chan, counts = utils.gapfill(in_chan,in_counts, config['NCHAN'])

    assert chan, counts == exp_chan, exp_counts



test_read_pxheader(xfmap)
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