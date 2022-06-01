struct NotImplementedException <: Exception end
struct DimNotImplementedException <: Exception end
struct InvalidDefinitionException <: Exception 
    reason::String
    InvalidDefinitionException() = new("")
    InvalidDefinitionException(reason::String) = new(reason)
end
Base.showerror(io::IO, e::NotImplementedException) = print(io, "The called feature is not implemented")
Base.showerror(io::IO, e::DimNotImplementedException) = print(io, "The called feature is not implemented for this dimension")
Base.showerror(io::IO, e::InvalidDefinitionException) = print(io, e.reason == "" ? "This is invalid by definition" : "This is invalid by definition because $(reason)")