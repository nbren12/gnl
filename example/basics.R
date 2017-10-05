

# colon operator

i = 1:10
# similar to range in python
i = seq(0, 9)

# R is column-major
matrix(i, nrow=2)

# R vector is kind of like python dict
v = c()

# multidimensional array
R = array(runif(1000), c(10,10,10))

# size in fortran is 
dim(R)

# some other info on basic R datatypes
# https://www.tutorialspoint.com/r/r_data_types.htm

# sorting
sortme <- runif(100)
order(sortme) 
sort(sortme)

# dplyr 
# uses
# 1. summarise
# 2. group_by


# for loop
for (i in 1:10) {
  print(i)
}

# if
if (TRUE) {
  print ("hello");
} else {
  print ("goodbye")
}
