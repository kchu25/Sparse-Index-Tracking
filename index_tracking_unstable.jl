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
        first "K_yes" rows in the feature matrix X; so we "should" see 
        the trained weight vector has non-zero entries mostly at the 
        first "K_yes" entries. Other entires should have values close 
        to zero
"""
function get_fake_data(K_yes=80, K_no=500, T=4000)
    # generate a feature matrix that generates the outcome using standard normal
    X_gen = randn((T, K_yes));
    # generate a feature matrix that does not generate the outcome using standard normal
    X_others = randn((T, K_no));    
    # generate the weight vector that generates the outcome using uniform distribution on [0,1]
    w_gen = rand(K_yes); w_gen = w_gen ./ sum(w_gen);
    # the labels
    rb = X_gen*w_gen;
    # make a feature matrix that contain both of the features 
    X = hcat(X_gen, X_others); 

    # pre-calculate the repeatedly used helper variables for ADMM update steps
    inv_XTX_plus_3I = inv(transpose(X)*X+3I);
    Xt = Transpose(X);
    inv_XTX_plus_3I_times_Xt = inv_XTX_plus_3I * Xt;
    T = size(X,1)   # T days
    N = size(X,2);  # N assets
    return rb, X, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T, N, w_gen
end

"""
This is same as above, except that we read the real data;
the real data is in the format "clean.csv" provided by Jason
"""
function get_real_data(fn::String)
    data = CSV.read(fn, DataFrame, delim=',');
    m = Matrix(data);
    # label and features
    rb = Float64.(m[1,5:end]);
    X = rotr90(Float64.(m[2:end,5:end]));    
    # pre-calculate the repeatedly used helper variables for ADMM update steps
    inv_XTX_plus_2I = inv(transpose(X)*X+2I);
    Xt = Transpose(X);
    inv_XTX_plus_2I_times_Xt = inv_XTX_plus_2I * Xt;
    T = size(X,1)  # T days
    N = size(X,2); # N assets
    return rb, X, inv_XTX_plus_2I, inv_XTX_plus_2I_times_Xt, T, N
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
    invXTX_p_2I_t_Xt::Matrix{Float64}           
    inv_XTX_p_2I::Matrix{Float64}    
    # weight vectpr
    w::Vector{Float64}
    w_gen::Union{Nothing, Vector{Float64}}      # ground truth weight vector
    # auxiliary variables
    v1::Vector{Float64}
    v2::Vector{Float64}
    v3::Vector{Float64}
    # dual variables (lagrange multipliers)
    u1::Vector{Float64}
    u2::Vector{Float64}
    u3::Vector{Float64}
    
    # this is the constructor
    function index_tracking(λ, ρ, rb, X, T, N, invXTX_p_2I_t_Xt, inv_XTX_p_2I, w_gen)
        # initialize the variables to be optimized by just using uniform distribution on [0,1]
        # the sparse vector
        w = rand(N); w = w ./ sum(w);
        # auxiliary variables (v) and lagrange multipliers (u)
        v1 = X*w;           u1 = rand(size(v1,1));
        v2 = copy(w);       u2 = rand(size(v2,1));
        v3 = copy(w);       u3 = rand(size(v3,1));
        # make the object
        new(λ,ρ,rb,X,T,N,λ/ρ,invXTX_p_2I_t_Xt,inv_XTX_p_2I,w,w_gen,v1,v2,v3,u1,u2,u3)
    end
end

# optimization  
function update_w!(q::index_tracking)
    q.w = q.invXTX_p_2I_t_Xt*(q.v1-q.u1) + q.inv_XTX_p_2I*(q.v2-q.u2) + q.inv_XTX_p_2I*(q.v3-q.u3);
end

function update_v1!(q::index_tracking)
    q.v1 = (q.rb + q.ρ*(q.X*q.w+q.u1)) ./ (1+q.ρ);
end

function update_v2!(q::index_tracking) # hard-threshold 
    w_plus_u2 = q.w + q.u2;
    w_plus_u2[abs.(w_plus_u2) .< sqrt(2*q.lmda_div_rho)] .= 0;
    q.v2 = w_plus_u2;
end

"""
Bisection 

Input:
    f: a scalat function
    lb: the initial lower bound
    ub: the initial upper bound
    ϵ: the tolerance parameter
Output:
    μ: a root of the equation f(x)=0
"""
function bisection(f, lb, ub, ϵ=1e-4)
    μ = nothing;
    if f(lb)*f(ub) > 0
        error("f(lb)f(ub)>0");
    end
    iter = 0;
    while ub-lb > ϵ
        μ = (ub+lb)/2; 
        iter += 1;
        if f(lb)*f(μ) > 0
            lb = μ;
        else
            ub = μ;
        end
        # println("iter: $iter, cur-sol: $μ");
    end
    return μ
end

function root_function_v3(w_plus_u3, mu)
    len = length(w_plus_u3);
    non_neg_term = w_plus_u3 - mu*ones(len); 
    non_neg_term = max.(non_neg_term, 0);
    return sum(ones(len) .* non_neg_term) - 1
end

function update_v3!(q::index_tracking)
    w_plus_u3 = q.w + q.u3;
    root_fcn(μ) = root_function_v3(w_plus_u3, μ);
    ub = 1e-5; 
    while root_fcn(ub) > 0 ub *=2 end;
    mu = bisection(root_fcn, 0, ub);
    w_plus_u3[w_plus_u3 .< mu] .= 0;
    q.v3 = w_plus_u3;
end

function update_lagrange!(q::index_tracking)
    q.u1 = q.u1 + q.X*q.w - q.v1;
    q.u2 = q.u2 + q.w - q.v2;
    q.u3 = q.u3 + q.w - q.v3;
end
# objective value
function objective(q::index_tracking)
    # this assumes that weight vector w is in the constrained set
    (norm(q.rb-q.X*q.w)^2)/q.T + q.λ*sum(q.w .> 0)
end
#######################################################################


################# to check result on simulated data, run this #########

# number of iterations
num_iter = 39;
# hyperparameters
rho = 1f0; lmda = 0.01; lmda_div_rho = lmda/rho;
# data and helper variables
rb, X, inv_XTX_plus_3I, inv_XTX_plus_3I_times_Xt, T, N, w_gen = get_fake_data();
# initialize our model
q = index_tracking(lmda, rho, rb, X, T, N, inv_XTX_plus_3I_times_Xt, inv_XTX_plus_3I, w_gen);

obj_values = [];

for _ = 1:num_iter
    update_w!(q);
    update_v1!(q);
    update_v2!(q);
    update_v3!(q);
    update_lagrange!(q)
    push!(obj_values, objective(q));
end

plot(1:length(obj_values), obj_values)
#######################################################################

################# to check result on real data, run this ##############

# file
fn = "clean.csv";
# number of iterations
num_iter = 10;
# hyperparameters
rho = 1f0; lmda = 0.01; lmda_div_rho = lmda/rho;
# data and helper variables
rb, X, invXTX_p_2I_t_Xt, inv_XTX_plus_2I_times_Xt, T, N = get_real_data(fn);
# initialize our model
q = index_tracking(lmda, rho, rb, X, T, N, inv_XTX_plus_2I_times_Xt, invXTX_p_2I_t_Xt, nothing);

obj_values = [];

for _ = 1:num_iter
    update_w!(q);
    update_v1!(q);
    update_v2!(q);
    update_v3!(q);
    update_lagrange!(q)
    push!(obj_values, objective(q));
end

plot(1:length(obj_values), obj_values)

#######################################################################
