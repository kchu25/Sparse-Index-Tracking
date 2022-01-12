"""
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
        println("iter: $iter, cur-sol: $μ");
    end
    return μ
end


# test
# sqrt_2(x) = x^2-2;
# bisection(sqrt_2, 1,2)