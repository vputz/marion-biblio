from pytest_bdd import scenario, given, when, then
from marion_biblio.bibliomatrix import OccurrenceMatrix, CooccurrenceType
import numpy
from pyrsistent import pvector
from marion_biblio.pyrsistent_helpers import pv_remove


@scenario('bibliomatrix.feature', 'test pvector2 deletion')
def test_pvector2_deletion():
    pass


@given('a pvector')
def sample_pvector():
    return pvector(['a', 'b', 'c', 'd'])


@then('check it deletes items')
def check_pvector_deletion(sample_pvector):
    v = pv_remove(sample_pvector, 'a', 'c')
    assert v == ['b', 'd']


@scenario('bibliomatrix.feature', 'column-prune a bibliomatrix')
def test_column_prune_bibliomatrix():
    pass


@given('a sample occurrence matrix')
def sample_occurrence_matrix():
    result = OccurrenceMatrix(
        numpy.array([[1, 1, 0, 1],
                     [1, 1, 1, 0],
                     [0, 0, 0, 1]]),
        ["paper_a", "paper_b", "paper_c"],
        ["author_a", "author_b", "author_c", "author_d"])
    return result


@given('it is column-pruned of small entries')
def column_pruned_occurrence_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.column_pruned(1)


@then('check its column-reduced form is correct')
def check_column_pruned_matrix(sample_occurrence_matrix,
                               column_pruned_occurrence_matrix):
    sm = sample_occurrence_matrix
    cp = column_pruned_occurrence_matrix
    print(cp)
    assert cp.is_consistent
    assert cp.rows == \
        sm.rows

    assert cp.columns == pv_remove(sm.columns, "author_c")
    print(cp)
    assert (cp.matrix ==
            numpy.array([[1, 1, 1],
                         [1, 1, 0],
                         [0, 0, 1]])).all()


@scenario('bibliomatrix.feature', 'row-prune a bibliomatrix')
def test_row_prune_bibliomatrix():
    pass


@given('it is row-pruned of small entries')
def row_pruned_occurrence_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.row_pruned(1)


@then('check its row-reduced form is correct')
def check_row_pruned_matrix(sample_occurrence_matrix,
                            row_pruned_occurrence_matrix):
    rp = row_pruned_occurrence_matrix
    sm = sample_occurrence_matrix
    print(rp)
    assert rp.is_consistent
    assert rp.columns == sm.columns
    assert rp.rows == sm.rows.delete(sm.rows.index('paper_c'))
    assert numpy.allclose(rp.matrix,
                          numpy.array([[1, 1, 0, 1],
                                       [1, 1, 1, 0]]))


@scenario('bibliomatrix.feature', 'test transposition')
def test_transposition():
    pass


@given('its transpose')
def sample_transposition(sample_occurrence_matrix):
    return sample_occurrence_matrix.transposed()


@then('check transpose is correct')
def check_transpose(sample_occurrence_matrix, sample_transposition):
    sm = sample_occurrence_matrix
    st = sample_transposition
    assert st.is_consistent
    assert st.rows == sm.columns
    assert st.columns == sm.rows
    print(st)
    assert numpy.allclose(st.matrix,
                          numpy.array(
                              [[1, 1, 0],
                               [1, 1, 0],
                               [0, 1, 0],
                               [1, 0, 1]]))


@scenario('bibliomatrix.feature', 'test cooccurrence calculations')
def test_cooccurrence_calcs():
    pass


@given('calculate its simple cooccurrence')
def simple_cooccurrence_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.column_cooccurrence()


@given('calculate its association index')
def association_index_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.column_cooccurrence(
        CooccurrenceType.association_index)


@given('calculate its cosine index')
def cosine_index_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.column_cooccurrence(
        CooccurrenceType.cosine_index)


@given('calculate its inclusion index')
def inclusion_index_matrix(sample_occurrence_matrix):
    return sample_occurrence_matrix.column_cooccurrence(
        CooccurrenceType.inclusion_index)


@then('check the cooccurrence matrix is correct')
def check_simple_cooccurrence(sample_occurrence_matrix,
                              simple_cooccurrence_matrix):
    sc = simple_cooccurrence_matrix
    assert sc.is_consistent
    assert numpy.allclose(sc.matrix,
                          numpy.array([[0, 2, 1, 1],
                                       [2, 0, 1, 1],
                                       [1, 1, 0, 0],
                                       [1, 1, 0, 0]]))


@then('check the association index is correct')
def check_association_index(sample_occurrence_matrix,
                            association_index_matrix):
    am = association_index_matrix
    assert am.is_consistent
    print(am)
    assert numpy.allclose(am.matrix,
                          numpy.array([[0,    0.5,  0.5,  0.25],
                                       [0.5,  0,    0.5,  0.25],
                                       [0.5,  0.5,    0,  0],
                                       [0.25, 0.25,   0,  0]]))


@then('check the cosine index is correct')
def check_cosine_index(sample_occurrence_matrix,
                       cosine_index_matrix):
    am = cosine_index_matrix
    assert am.is_consistent
    print(am)
    assert numpy.allclose(
        am.matrix,
        numpy.array([[0, 1, 1/numpy.sqrt(2), 0.5],
                     [1, 0, 1/numpy.sqrt(2), 0.5],
                     [1/numpy.sqrt(2), 1/numpy.sqrt(2), 0, 0],
                     [0.5, 0.5, 0, 0]]))


@then('check the inclusion index is correct')
def check_inclusion_index(sample_occurrence_matrix,
                          inclusion_index_matrix):
    am = inclusion_index_matrix
    assert am.is_consistent
    print(am)
    assert numpy.allclose(am.matrix,
                          numpy.array([[0, 1, 1, 0.5],
                                       [1, 0, 1, 0.5],
                                       [1, 1, 0, 0],
                                       [0.5, 0.5, 0, 0]]))
