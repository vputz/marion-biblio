from pytest_bdd import scenario, given, then, parsers
from marion_biblio import wos_reader
import tempfile
import os
import shutil


def samplefile_path(filename):
    to_testfiles = os.path.abspath(
        os.path.join(os.path.split(__file__)[0], "..", "test_data"))
    path = os.path.join(to_testfiles, filename)
    return path


@scenario('wos_reader.feature', 'Convert WOS data')
def test_wos_conversion():
    pass


@given(parsers.parse('the test file {tsv_filename}'))
def sample_wos_reader(tsv_filename):
    result = wos_reader.WosReader(samplefile_path(tsv_filename))
    return result


@given(parsers.parse('create the WOS file {h5_filename}'))
def sample_h5_file(request, sample_wos_reader, h5_filename):
    tempworkingdir = tempfile.mkdtemp()
    os.mkdir(os.path.join(tempworkingdir, "transactions"))

    def fin():
        shutil.rmtree(tempworkingdir)
    request.addfinalizer(fin)

    h5_filename = os.path.join(tempworkingdir, "sample_wos.h5")

    wos_reader.make_pytable(
        sample_wos_reader,
        h5_filename,
        "Irwin"
    )

    result = wos_reader.Wos_h5_reader(h5_filename)
    return result


@then('ensure wos/h5 data are correct')
def check_wos_data(sample_h5_file):
    print([row['title'] for row in sample_h5_file.h5.root.papers])
    assert len(sample_h5_file.all_authors()) == 14
    assert b'Zaid, I' in sample_h5_file.all_authors()
    assert b'Non-Gauss Athermal Fluctuations in Bacterial Bath' \
        in [x['title'] for x in sample_h5_file.papers]
