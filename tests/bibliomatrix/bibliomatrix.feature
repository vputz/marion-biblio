Feature: Implement bibliomatrices
  Implement the cooccurrence functionality in a class to simplify
  bookkeeping and testing.

Scenario: test pvector2 deletion
  Given a pvector
  Then check it deletes items

Scenario: column-prune a bibliomatrix
  Given a sample occurrence matrix
  And it is column-pruned of small entries
  Then check its column-reduced form is correct

Scenario: row-prune a bibliomatrix
  Given a sample occurrence matrix
  And it is row-pruned of small entries
  Then check its row-reduced form is correct

Scenario: test cooccurrence calculations
  Given a sample occurrence matrix
  And calculate its simple cooccurrence
  And calculate its association index
  And calculate its cosine index
  And calculate its inclusion index
  Then check the cooccurrence matrix is correct
  And check the association index is correct
  And check the cosine index is correct
  And check the inclusion index is correct

Scenario: test transposition
  Given a sample occurrence matrix
  And its transpose
  Then check transpose is correct


