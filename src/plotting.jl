
@recipe function plot(s::BSpline, x = linspace(s.knots.min, s.knots.max, 500) )
    #markershape --> :auto        # if markershape is unset, make it :auto
    #markercolor :=  customcolor  # force markercolor to be customcolor
    xrotation   --> 0#45           # if xrotation is unset, make it 45
    zrotation   --> 90           # if zrotation is unset, make it 90
    legend --> :none
    xlabel --> "X Value"
    ylabel --> "Y Value"
    
    x, evaluate(s, x)
end

@recipe function plot(s::BSpline, ::Val{∂}, x = linspace(s.knots.min, s.knots.max, 500)) where ∂
    #markershape --> :auto        # if markershape is unset, make it :auto
    #markercolor :=  customcolor  # force markercolor to be customcolor
 #   xrotation   --> 0#45           # if xrotation is unset, make it 45
 #   zrotation   --> 90           # if zrotation is unset, make it 90
    legend --> :none
    xlabel --> "X Value"
    ylabel --> "Derivative #$∂"                   # return the arguments (input data) for the next recipe
    v = evaluate(s, x, Val{∂}())
    x, v
end

@recipe function plot(s::BSpline, ::Val{:phase}, x::AbstractVector = linspace(s.knots.min, s.knots.max, 500))
    #markershape --> :auto        # if markershape is unset, make it :auto
    #markercolor :=  customcolor  # force markercolor to be customcolor
#    xrotation   --> 0#45           # if xrotation is unset, make it 45
#    zrotation   --> 90           # if zrotation is unset, make it 90
    legend --> :none                   # return the arguments (input data) for the next recipe
    seriestype --> :path
    xlabel --> "Velocity"
    ylabel --> "Acceleration"
    d1 = evaluate(s, x, Val{1}())
    d2 = evaluate(s, x, Val{2}())
    if any(isinf.(d2))
        sub = .!isinf.(d2)
        d1 = d1[sub]
        d2 = d2[sub]
    end
    d1, d2
end
