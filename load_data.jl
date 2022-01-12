using DataFrames, CSV, LinearAlgebra, Plots



data = CSV.read("clean.csv", DataFrame, delim=',');
# m=Matrix(data);
# names = m[:,2];
# ticker = m[:,3];

