struct Rect
    lx::Float64
    ly::Float64
    ux::Float64
    uy::Float64
    function Rect(lx::Real, ly::Real, ux::Real, uy::Real)
        if lx > ux || ly > uy
            throw(ErrorException("Invalid rectangle"))
        end
        new(lx, ly, ux, uy)
    end
end

EmptyRect = Rect(0, 0, 0, 0)


Base.isempty(a::Rect) = (a.lx == a.ux) || (a.ly == a.uy)
Base.isempty(a::AbstractVector{Rect}) = all(isempty.(a))

function isdisjoint(a::Rect, b::Rect)::Bool
    return (a.ux <= b.lx) || (a.lx >= b.ux) || (a.ly >= b.uy) || (a.uy <= b.ly)
end

function isdisjoint(a::AbstractVector{Rect}, b::Rect)::Bool
    for h ∈ a
        if !isdisjoint(h, b)
            return false
        end
    end
    return true
end

function isdisjoint(a::AbstractVector{Rect}, b::AbstractVector{Rect})::Bool
    for bi ∈ b
        if !isdisjoint(a, bi)
            return false
        end
    end
    return true
end

function isdisjoint(a::Rect, b::AbstractVector{Rect})::Bool
    return isdisjoint([a], b)
end

## if A contains B, i.e. B ⊆ A => B \ A = ∅
contains(a::Rect, b::Rect)::Bool = isempty(difference(b, a))
contains(a::AbstractVector{Rect}, b::Rect)::Bool = isempty(difference(b, a))
contains(a::Rect, b::AbstractVector{Rect})::Bool = isempty(difference(b, a))
contains(a::AbstractVector{Rect}, b::AbstractVector{Rect})::Bool = isempty(difference(b, a))

function difference(a::Rect, b::Rect)::Vector{Rect}
    # if a ∩ B = ∅, then A \ B = A
    if isdisjoint(a, b)
        return [a]
    else
        result = Vector{Rect}(undef, 0)
        if a.lx < b.lx 
            insert!(result, 1, Rect(a.lx, a.ly, b.lx, a.uy))
            if a.ly < b.ly
                insert!(result, 1, Rect(b.lx, a.ly, min(a.ux, b.ux), b.ly))
            end
            if a.uy > b.uy
                insert!(result, 1, Rect(b.lx, b.uy, min(a.ux, b.ux), a.uy))
            end
        else 
            if a.ly < b.ly
                insert!(result, 1, Rect(a.lx, a.ly, min(a.ux, b.ux), b.ly))
            end
            if a.uy > b.uy
                insert!(result, 1, Rect(a.lx, b.uy, min(a.ux, b.ux), a.uy))
            end
        end
        if a.ux > b.ux
            insert!(result, 1, Rect(b.ux, a.ly, a.ux, a.uy))
        end
        return result
    end
end

function difference(a::AbstractVector{Rect}, b::Rect)::Vector{Rect}
    result = Vector{Rect}(undef, 0)
    for r ∈ a
        union!(result, difference(r, b))
    end
    return result
end

function difference(a::AbstractVector{Rect}, b::AbstractVector{Rect})::Vector{Rect}
    result = deepcopy(a)
    for bi ∈ b
        result = difference(result, bi)
    end
    return result
end

function difference(a::Rect, b::AbstractVector{Rect})::Vector{Rect}
    return difference([a], b)
end

function intersection(a::Rect, b::Rect)
    if isdisjoint(a, b)
        return EmptyRect
    else
        return Rect(max(a.lx, b.lx), max(a.ly, b.ly), min(a.ux, b.ux), min(a.uy, b.uy))
    end
end
function intersection(a::AbstractVector{Rect}, b::Rect)
    result = Vector{Rect}(undef, 0)
    for r ∈ a
        inters = intersection(r, b)
        if !isempty(inters)
            insert!(result,1, inters)
        end
    end
    return result
end

function intersection(a::AbstractVector{Rect}, b::AbstractVector{Rect})
    result = Vector{Rect}(undef, 0)
    for bi ∈ b
        union!(result, intersection(a, bi))
    end
    return result
end
function intersection(a::Rect, b::AbstractVector{Rect})
    return intersection([a], b)
end