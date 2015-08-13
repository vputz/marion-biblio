Feature: Convert WOS data
  Take downloaded WOS tab separated values and 
  compile them into an intermediate HDF5 format

Scenario: Convert WOS data
  Given the test file irwin.tab
  And create the WOS file irwin.h5
  Then ensure wos/h5 data are correct
