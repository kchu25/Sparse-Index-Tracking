using DataFrames, CSV, LinearAlgebra, Plots

################# helper functions to get the data ####################
"""
Input: 
    K_yes: number of features that actually generate the outcome
    K_no: number of features that doesn't generate the outcome
    T: days
Output:
    rb: the outcomes (index)
    X: feature matrix (stocks)
    inv_XTX_plus_3I: helper variable used in update step 1
    Xt: transpose of the feature matrix X
    inv_XTX_plus_3I_times_Xt: another helper variable used in update step 1
    T: days
    N: number of assets
    w_gen: the groud truth weight vector

Note: 
    1. By "outcome" I mean the actual index
    2. the features (stocks) that generate the weight vector are the 
        first "K_yes" columns in the feature matrix X; so we "should" see 
        the trained weight vector has non-zero entries mostly at the 
        first "K_yes" entries. Other entires should have values close 
        to zero
"""
function get_fake_data(K_yes=80, K_no=500, T=4000, train_test_ratio=0.8)
    # generate a feature matrix that generates the outcome using standard normal
    X_gen =  randn((T, K_yes));
    # generate a feature matrix that does not generate the outcome using standard normal
    X_others = randn((T, K_no));    
    # generate the weight vector that generates the outcome using uniform distribution on [0,1]
    w_gen = rand(K_yes); w_gen = w_gen ./ sum(w_gen);
    # the labels
    rb = X_gen*w_gen;
    # make a feature matrix that contain both of the features 
    X = hcat(X_gen, X_others); 
    T = size(X,1);  # days
    N = size(X,2);  # N assets

    @assert 0 ≤ train_test_ratio ≤ 1 "train_test_ratio must be in (0,1)"
    # pre-calculate the repeatedly used helper variables for ADMM update steps
    num_train_entries = Int(floor(train_test_ratio*T));
    X_train = X[1:num_train_entries,:]; X_test = X[num_train_entries+1:end,:];
    rb_train = rb[1:num_train_entries]; rb_test = rb[num_train_entries+1:end];
    inv_XTX_plus_3I = inv(transpose(X_train)*X_train+3I);
    Xt = Transpose(X_train);
    inv_XTX_plus_3I_times_Xt = inv_XTX_plus_3I * Xt;
    T_train = size(X_train,1);  # T days in the training set
    T_test = size(X_train,1);   # T days in the test set
    return rb_train, rb_test, X_train, X_test, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T_train, T_test, N, w_gen
end

"""
This is same as above, except that we read the real data;
the real data is in the format "clean.csv" provided by Jason
"""
function get_real_data(fn::String, train_test_ratio=0.8)
    data = CSV.read(fn, DataFrame, delim=',');
    m = Matrix(data);
    # label and features
    rb = Float64.(m[1,5:end]);
    X = rotr90(Float64.(m[2:end,5:end]));    
    T = size(X,1); # days
    N = size(X,2); # N assets
    # pre-calculate the repeatedly used helper variables for ADMM update steps
    num_train_entries = Int(floor(train_test_ratio*T));
    X_train = X[1:num_train_entries,:]; X_test = X[num_train_entries+1:end,:];
    rb_train = rb[1:num_train_entries]; rb_test = rb[num_train_entries+1:end];
    inv_XTX_plus_3I = inv(transpose(X_train)*X_train+3I);
    Xt = Transpose(X_train);
    inv_XTX_plus_3I_times_Xt = inv_XTX_plus_3I * Xt;
    T_train = size(X_train,1);  # T days in the training set
    T_test = size(X_train,1);   # T days in the test set
    
    return rb_train, rb_test, X_train, X_test, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T_train, T_test, N
end
#######################################################################

################# ADMM ################################################
# the model; initialize it as a structure (like how we do OOP -- e.g. class in python)
mutable struct index_tracking
    # hyperparameters
    λ::Float64                                  # sparsity parameter
    ρ::Float64                                  # penalty parameter    
    # data
    rb::Vector{Float64}                         # index
    X::Matrix{Float64}                          # feature matrix (stock)
    T::Integer                                  # days
    N::Integer                                   # number of assets
    # helper variables 
    lmda_div_rho::Float64                       # λ/ρ
    invXTX_p_3I_t_Xt::Matrix{Float64}           
    inv_XTX_p_3I::Matrix{Float64}    
    # weight vectpr
    w::Vector{Float64}
    w_gen::Union{Nothing, Vector{Float64}}      # ground truth weight vector
    # auxiliary variables
    v1::Vector{Float64}
    v2::Vector{Float64}
    v3::Vector{Float64}
    v4::Vector{Float64}
    # dual variables (lagrange multipliers)
    u1::Vector{Float64}
    u2::Vector{Float64}
    u3::Vector{Float64}
    u4::Vector{Float64}
    
    # this is the constructor
    function index_tracking(λ, ρ, rb, X, T, N, invXTX_p_3I_t_Xt, inv_XTX_p_3I, w_gen)
        # initialize the variables to be optimized by just using uniform distribution on [0,1]
        # the sparse vector
        w = rand(N); w = w ./ sum(w);
        # auxiliary variables (v) and lagrange multipliers (u)
        v1 = X*w;           u1 = rand(size(v1,1));
        v2 = copy(w);       u2 = rand(size(v2,1));
        v3 = copy(w);       u3 = rand(size(v3,1));
        v4 = copy(w);       u4 = rand(size(v4,1));
        # make the object
        new(λ,ρ,rb,X,T,N,λ/ρ,invXTX_p_3I_t_Xt,inv_XTX_p_3I,w,w_gen,v1,v2,v3,v4,u1,u2,u3,u4)
    end
end

# optimization  
function update_w!(q::index_tracking)
    q.w = q.invXTX_p_3I_t_Xt*(q.v1-q.u1) + q.inv_XTX_p_3I*(q.v2-q.u2) + q.inv_XTX_p_3I*(q.v3-q.u3) + q.inv_XTX_p_3I*(q.v4-q.u4);
end

function update_v1!(q::index_tracking)
    q.v1 = (q.rb + q.ρ*(q.X*q.w+q.u1)) ./ (1+q.ρ);
end

function update_v2!(q::index_tracking) # hard-threshold 
    w_plus_u2 = q.w + q.u2;
    w_plus_u2[abs.(w_plus_u2) .< sqrt(2*q.lmda_div_rho)] .= 0;
    q.v2 = w_plus_u2;
end

function update_v3!(q::index_tracking) # proximal on the box
    w_plus_u3 = q.w + q.u3;
    w_plus_u3[w_plus_u3 .< 0] .= 0;
    w_plus_u3[w_plus_u3 .> 1] .= 0;
    q.v3 = w_plus_u3;
end

function update_v4!(q::index_tracking) # proximal on the simplex
    w_plus_u4 = q.w + q.u4;
    sum_w_plus_u4_minus_1_div_N = (sum(w_plus_u4)-1)/length(q.w);
    q.v4 = w_plus_u4 .- sum_w_plus_u4_minus_1_div_N;
end

function update_lagrange!(q::index_tracking)
    q.u1 = q.u1 + q.X*q.w - q.v1;
    q.u2 = q.u2 + q.w - q.v2;
    q.u3 = q.u3 + q.w - q.v3;
    q.u4 = q.u4 + q.w - q.v4;
end
# objective value
function objective(q::index_tracking)
    # this assumes that weight vector w is in the constrained set
    (norm(q.rb-q.X*q.w)^2)/q.T + q.λ*sum(q.w .> 0)
end
#######################################################################


################# to check result on simulated data, run this #########

# number of iterations
num_iter = 10;
# hyperparameters
rho = 1f0; lmda = 0.01; lmda_div_rho = lmda/rho;
# data and helper variables
rb_train, rb_test, X_train, X_test, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T_train, T_test, N, w_gen = get_fake_data();
# initialize our model

q = index_tracking(lmda, rho, rb_train, X_train, T_train, N, inv_XTX_plus_3I_times_Xt, inv_XTX_plus_3I, w_gen);

obj_values = [];

for _ = 1:num_iter
    update_w!(q);
    update_v1!(q);
    update_v2!(q);
    update_v3!(q);
    update_v4!(q);
    update_lagrange!(q)
    push!(obj_values, objective(q));
end

# plot the objective values during the iteration
plot(1:length(obj_values), obj_values, ylabel="objective value", xlabel="# iterations",label=nothing)


# we "trim off" the weights in w that are too small, and renormalize
cut_off = 0.001;
num_entries_after_cutoff = sum(q.w .> cut_off);
println("number of entries in w bigger than cut_off: $num_entries_after_cutoff");
q.w[q.w .< cut_off] .= 0; 
q.w = q.w ./ sum(q.w);

plot!(1:length(rb_test), X_test * q.w, label="predicted")
plot!(1:length(rb_test), rb_test, label="Test set")
plot!(xlabel="days", ylabel="index value")
#######################################################################

################# to check result on real data, run this ##############

# file
fn = "sp500.csv";
# number of iterations
num_iter = 50;
# hyperparameters
rho = 1f0; lmda = 1f0; lmda_div_rho = lmda/rho;
# data and helper variables
rb_train, rb_test, X_train, X_test, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T_train, T_test, N = get_real_data(fn);
# initialize our model
q = index_tracking(lmda, rho, rb_train, X_train, T_train, N, inv_XTX_plus_3I_times_Xt, inv_XTX_plus_3I, nothing);

obj_values = [];

for _ = 1:num_iter
    update_w!(q);
    update_v1!(q);
    update_v2!(q);
    update_v3!(q);
    update_v4!(q);
    update_lagrange!(q)
    push!(obj_values, objective(q));
end


# plot the objective values during the iteration
plot(1:length(obj_values), obj_values)

# we "trim off" the weights in w that are too small, and renormalize
cut_off = 0.001;
num_entries_after_cutoff = sum(q.w .> cut_off);
println("number of entries in w bigger than cut_off: $num_entries_after_cutoff");
q.w[q.w .< cut_off] .= 0; 
q.w = q.w ./ sum(q.w);

plot(1:length(rb_test), X_test * q.w, label="predicted")
plot!(1:length(rb_test), rb_test, label="actual")
plot!(xlabel="days", ylabel="index value")
#######################################################################
