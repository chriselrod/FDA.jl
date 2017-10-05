

### Okay, for penalty, you can forgo calculation of G altogether.
### J = β'Sβ, S = G'WG, therefore J = (Gβ)'WGβ. You have already implemented the method for calculating Gβ!!! Alternatively, should be straightforward to do with fillΦ! as well.
### Question is -- what is easiest? Actual calculation of Gβ would aid solving via linear algebra.
###
function BSplinePenalty(s::BSpline, deriv::Val{d}) where d
    for (lb, ub, h) ∈ intervals(s.knots)

    end

end



###Use exp(λ)*(max-min)^(2m-1) as penalty coefficient.
