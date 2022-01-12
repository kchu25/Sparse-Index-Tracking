
function update_w(v1, u1, v2, u2, v3, u3, v4, u4)
    return inv_XTX_plus_3I_times_Xt*(v1-u1) + inv_XTX_plus_3I*(v2-u2) + inv_XTX_plus_3I*(v3-u3)+inv_XTX_plus_3I*(v4-u4);
end

function update_v1(w, u1)
    return (rb + rho*(X*w+u1)) ./ (1+rho);
end

function update_v2(w, u2, lmda_div_rho) # hard-threshold 
    w_plus_u2 = w + u2;
    w_plus_u2[abs.(w_plus_u2) .< sqrt(2*lmda_div_rho)] .= 0;
    return w_plus_u2
end

function update_v3(w, u3)
    w_plus_u3 = w + u3;
    w_plus_u3[w_plus_u3 .< 0] .= 0;
    w_plus_u3[w_plus_u3 .> 1] .= 0;
    return w_plus_u3
end

function update_v4(w, u4)
    w_plus_u4 = w + u4;
    sum_w_plus_u4_minus_1_div_N = (sum(w_plus_u4)-1)/length(w);
    return w_plus_u4 .- sum_w_plus_u4_minus_1_div_N;
end

# function update_v3(w, u3)
#     w_plus_u3 = w + u3;
#     root_fcn(μ) = root_function_v3(w_plus_u3, μ);
#     ub = 1e-5; 
#     while root_fcn(ub) > 0 ub *=2 end;
#     mu = bisection(root_fcn, 0, ub);
#     w_plus_u3[w_plus_u3 .< mu] .= 0;
#     return w_plus_u3
# end


# # check that root fcn in terms of μ is strictly decreasing
# # plot(0:0.001:1, root_fcn.(0:0.001:1))

# function root_function_v3(w_plus_u3, mu)
#     len = length(w_plus_u3);
#     non_neg_term = w_plus_u3 - mu*ones(len); 
#     non_neg_term = max.(non_neg_term, 0);
#     return sum(ones(len) .* non_neg_term) - 1
# end

