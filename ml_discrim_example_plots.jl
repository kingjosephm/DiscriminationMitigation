
using Distributions
using DataFrames
using Plots
using GLM
using Parameters


function draw_from_dgp(shocks::NamedTuple{(:σ²_y, :σ²_W, :class_prob), Tuple{Float64, Float64, Float64}},
                       coeffs::NamedTuple{(:γ₀, :γ_C, :α₀, :α₁, :β₀, :β₁), Tuple{Float64,Float64,Float64,Float64,Float64,Float64}} )
    @unpack σ²_y, σ²_W, class_prob = shocks
    @unpack γ₀, γ_C, α₀, α₁, β₀, β₁ =  coeffs

    # uncaused causes (shocks)
    e_yᵢ  = rand(Normal(0, σ²_y^0.5))
    e_Wᵢ  = rand(Normal(0, σ²_W^0.5))
    C1ᵢ = rand(Binomial(1, class_prob))
    C0ᵢ = 1 - C1ᵢ

    # caused varibles:
    Wᵢ = γ₀*C0ᵢ    + γ_C*C1ᵢ   + e_Wᵢ # W is a linear function of class and shocks
    yᵢ = α₀*C0ᵢ    + α₁ *C1ᵢ    + # Y is a linear function of class, W, and shocks.
         β₀*C0ᵢ*Wᵢ + β₁ *C1ᵢ*Wᵢ +
         e_yᵢ

    dataᵢ = [yᵢ C0ᵢ C1ᵢ Wᵢ]
    return dataᵢ
end



function simulate_df(N::Integer, draw::Function)
    M = Array{Float64,2}(undef, N, 4)
    for i in 1:N
        M[i,:] = draw()
    end
    df = DataFrame(M)
    rename!(df, [:Y, :C0, :C1, :W1])
    return df
end


struct ClassWeights
    class_weights::NamedTuple
end
function ClassWeights(class_weights::NamedTuple,df::DataFrame)
    @assert all(𝑉 ∈ names(df) for 𝑉 in keys(class_weights))  "class weight names must be variables of dataframe"
    var_is_a_dummy(𝑉, df) = all((df[:, 𝑉] .== 1) .| (df[:, 𝑉] .== 0))
    @assert all(var_is_a_dummy(𝑉, df) for 𝑉 in keys(class_weights)) "class weight names must refer to only dummy variables"
    @assert sum(values(class_weights)) == 1 "class weights must sum to 1"
    ClassWeights(class_weights)
end


import GLM.predict
function predict(fit::StatsModels.TableRegressionModel, cw::ClassWeights)
    coeffs = fit.model.pp.beta0
    rhs_vars = [fit.mf.f.rhs.terms[k].sym for k in 2:(length(coeffs)+1)] # constant is for 1, but estimate without constant
    pseudo_df(𝑉::Symbol) = 𝑉 ∈ keys(cw.class_weights) ? cw.class_weights[𝑉] : df[:, 𝑉]
    y_hat = 0
    for (k, 𝑉ₖ) in enumerate(rhs_vars )
        y_hat = y_hat .+ (pseudo_df(𝑉ₖ) .* coeffs[k])
    end
    return y_hat
end

# plot three predictions
function plot_three(shock_parameters, coefficients)
    dgp_draw() = draw_from_dgp(shock_parameters, coefficients)
    N = 1000
    df = simulate_df(N, dgp_draw)
    frml_direct = @formula(Y ~ 0 + C0 + C1 + W1*C0 + W1)
    frml_proxy = @formula(Y ~ 1 + W1)

    fit_direct = fit(LinearModel, frml_direct, df)
    fit_proxy = fit(LinearModel, frml_proxy, df)

    plt_direct = plot()
    scatter!(plt_direct, df[df.C1 .== 1.0, :W1], df[df.C1 .== 1.0, :Y],
             markersize=1.5, markershape=:circle, color = :blue)
    scatter!(plt_direct, df[df.C0 .== 1.0, :W1], df[df.C0 .== 1.0, :Y],
             markersize=1.5, markershape=:cross, color = :red)
    plot!(plt_direct, df.W1[df.C1 .== 1], predict(fit_direct)[df.C1 .== 1], color = :blue)
    plot!(plt_direct, df.W1[df.C0 .== 1], predict(fit_direct)[df.C0 .== 1], color = :red)

    plt_proxy = plot()
    scatter!(plt_proxy, df[df.C1 .== 1.0, :W1], df[df.C1 .== 1.0, :Y],
             markersize=1.5, markershape=:circle, color = :blue)
    scatter!(plt_proxy, df[df.C0 .== 1.0, :W1], df[df.C0 .== 1.0, :Y],
             markersize=1.5, markershape=:cross, color = :red)
    plot!(plt_proxy, df.W1[df.C1 .== 1], predict(fit_proxy)[df.C1 .== 1], color = :blue)
    plot!(plt_proxy, df.W1[df.C0 .== 1], predict(fit_proxy)[df.C0 .== 1], color = :red)
    plot!(plt_proxy, df.W1[df.C1 .== 1], predict(fit_direct)[df.C1 .== 1], color = :orange)
    plot!(plt_proxy, df.W1[df.C0 .== 1], predict(fit_direct)[df.C0 .== 1], color = :orange)

    cw = ClassWeights((C1=0.5, C0=0.5), df)
    plt_mitigated = plot()
    scatter!(plt_mitigated, df[df.C1 .== 1.0, :W1], df[df.C1 .== 1.0, :Y],
             markersize=1.5, markershape=:circle, color = :blue)
    scatter!(plt_mitigated, df[df.C0 .== 1.0, :W1], df[df.C0 .== 1.0, :Y],
             markersize=1.5, markershape=:cross, color = :red)
    plot!(plt_mitigated, df.W1, predict(fit_direct, cw), color = :purple)
    return plt_direct, plt_proxy, plt_mitigated
end

# case 1
shock_parameters = (σ²_y=0.2, σ²_W=1.0, class_prob=0.5)
coefficients = (γ₀=4.0, γ_C=6.0, α₀=0.0, α₁=1.5, β₀=1.0, β₁=-1.0)
plt1, plt2, plt3 = plot_three(shock_parameters, coefficients)

# case 1
shock_parameters = (σ²_y=0.2, σ²_W=1.0, class_prob=0.5)
coefficients = (γ₀=4.0, γ_C=10.0, α₀=0.0, α₁=5.0, β₀=1.0, β₁=1.0)
class_corr_with_W_but_no_ovb__dgp() = draw_from_dgp(shock_parameters, coefficients)

plt = plot()
scatter!(plt, df[df.C1 .== 1.0, :W1], df[df.C1 .== 1.0, :Y],
         markersize=1.5, markershape=:circle, color = :blue)
scatter!(plt, df[df.C0 .== 1.0, :W1], df[df.C0 .== 1.0, :Y],
         markersize=1.5, markershape=:cross, color = :red)
plot!(plt, df.W1[df.C1 .== 1], predict(fit1)[df.C1 .== 1], color = :blue)
plot!(plt, df.W1[df.C0 .== 1], predict(fit1)[df.C0 .== 1], color = :red)

model1_cw = ClassWeights((C1=0.5, C0=0.5), df)


predict(fit1, model1_cw)

plot!(plt, df.W1, predict(fit1, model1_cw), color = :purple)
