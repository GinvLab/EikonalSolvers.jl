
#
# MIT License
# Copyright (c) 2019 Andrea Zunino
# 

#######################################################
##               BinHeap                             ## 
#######################################################
"""
    Binary heaps (min/max)

     IMPORTANT: make sure the handle is unique!
      Check is commented out because slows things down...

"""
module BinHeaps


export BinHeapMax,BinHeapMin
export topval_heap,build_maxheap!,build_minheap!,insert_maxheap!,
    insert_minheap!,update_node_maxheap!,
    update_node_minheap!,pop_minheap!,pop_maxheap!
    

##=============================================================

mutable struct BinHeapMax
    Nmax::Int64
    Nh::Int64
    nodes::Array{Float64,1}
    handles::Array{Int64,1}
    idxhan::Array{Int64,1}
end


mutable struct BinHeapMin
    Nmax::Int64
    Nh::Int64
    nodes::Array{Float64,1}
    handles::Array{Int64,1}
    idxhan::Array{Int64,1}
end

##=============================================================

index_parent(i::Integer) = div(i,2) 

index_leftchild(i::Integer) = 2*i ## shift bits???

index_rightchild(i::Integer) = 2*i+1 ## shift bits???

##=============================================================


##=============================================================
##            General stuff
##=============================================================

function swap_nodes_heap!(h::Union{BinHeapMax,BinHeapMin},p::Integer,q::Integer)
    #                     h::{Union{BinHeapMax,BinHeapMin} means:
    #                       call if h is either of the two types
    # temporary copy
    ptmp_node = h.nodes[p]
    ptmp_handle = h.handles[p]
    ptmp_idxhan = h.idxhan[p]
    # swap last and first node
    h.nodes[p] = h.nodes[q]
    h.nodes[q] = ptmp_node
    # swap handles too
    h.handles[p] = h.handles[q]
    h.handles[q] = ptmp_handle
    # swap idx of handles too
    h.idxhan[h.handles[p]] = p
    h.idxhan[h.handles[q]] = q

    # @show  h.handles[p],h.idxhan[p]
    # @show  h.handles[q],h.idxhan[q]
    
    return   
end

##=============================================================

function topval_heap(h::Union{BinHeapMax,BinHeapMin})
    return h.nodes[1],h.handles[1]
end

##=============================================================



##=============================================================
##               MAX stuff
##=============================================================

function build_maxheap!(values::Array{Float64,1},Nmax::Integer,
                         handles::Array{Int64,1})
    @assert size(values,1)==size(handles,1)
    ## Init the structure 
    ## Nmax,Nh,nodes,handles
    Nh=size(values,1)
    ## Init big arrays
    ##Array{Float64}(0),10,Array{Int64}(0)
    values2  = zeros(Float64,Nmax)
    handles2 = zeros(Int64,Nmax)
    idxhan2 = zeros(Int64,Nmax)

    ## copy to avoid modifying input arrays!!
    values2[1:Nh]  = copy(values)
    handles2[1:Nh] = copy(handles)
    h = BinHeapMax(Nmax,Nh,values2,handles2,idxhan2)

    ## heapify the structure
    sta = div(h.Nh,2)
    for i=sta:-1:1 ## backwards
        max_heapify!(h,i)
    end

    ## get all the addresses of the handles
    for i=1:h.Nh
        h.idxhan[h.handle[i]] = i
    end
    
    return h
end

##=============================================================

function max_heapify!(h::BinHeapMax,i::Int64)
    ## Go DOWN the tree (max heap)...
    # get children indices
    l = index_leftchild(i)
    r = index_rightchild(i)
    ## Introduction to algorithms, p. 154
    if (l <= h.Nh) && (h.nodes[l]>h.nodes[i])
        # largest on left
        largest = l
    else
        # largest on i
        largest = i
    end
    if (r <= h.Nh) && (h.nodes[r]>h.nodes[largest])
        # largest on right
        largest = r
    end
    if largest!=i
        # swap the nodes
        swap_nodes_heap!(h,i,largest)
        # keep going
        max_heapify!(h,largest)
    end
    return
end

##=============================================================

function update_node_maxheap!(h::BinHeapMax,val::Float64,handle::Int64)
    ## find index of node given handle
    # idxh = find(h.handles.==handle)
    # @assert size(idxh,1)<=1      
    ## faster than Julia find...
    # idxh = 0
    # @inbounds for l=1:h.Nh
    #     if h.handles[l]==handle
    #         idxh = l
    #         break
    #     end
    # end
    idxh = h.idxhan[handle]

    i = idxh #[1] # from 1D array to scalar
    h.nodes[i] = val

    # GO UP THE TREE: compare with parents...
    @inbounds while (i>1) && (h.nodes[index_parent(i)]<h.nodes[i])
        ## swap node with parent
        swap_nodes_heap!(h,i,index_parent(i))
        i = index_parent(i)
    end

    # GO DOWN THE TREE: compare with children...
    if i<h.Nh
        max_heapify!(h,i)
    end
    return
end

##=============================================================

function insert_maxheap!(h::BinHeapMax,val::Float64,handle::Int64)
    ## Go UP the tree (max heap) from the bottom...
    # extend heap
    @assert h.Nmax>(h.Nh+1)
    ## IMPORTANT: make sure handle is unique
    ## commented because slows things down
    #   @assert ~(handle in h.handles[1:h.Nh])
    # resize
    h.Nh = h.Nh+1
    # add the new node at the bottom
    h.nodes[h.Nh] = val
    h.handles[h.Nh] = handle # set handle too
    h.idxhan[handle] = h.Nh # set idx of handle too

    # start from bottom right leaf
    i = h.Nh
    # compare with parents...
    @inbounds while (i>1) && (h.nodes[index_parent(i)]<h.nodes[i])
        ## swap node with parent
        swap_nodes_heap!(h,i,index_parent(i))
        i = index_parent(i)
    end
    return
end

##=============================================================

function pop_maxheap!(h::BinHeapMax)
    # save handle and values of top to return them
    poppedtophandle = h.handles[1]
    poppedval = h.nodes[1]
    # move last elem to top
    h.nodes[1] = h.nodes[h.Nh]
    h.handles[1] = h.handles[h.Nh]
    h.idxhan[h.handles[h.Nh]] = 1 # set idx of handle too
    ## set out of Nh nodes to 0
    h.nodes[h.Nh] = 0.0
    h.handles[h.Nh] = 0
    ## set to 0 the address of the REMOVED node (poppedtophandle)
    h.idxhan[poppedtophandle] = 0
    # shorten heap
    h.Nh = h.Nh-1
    # swap the root [1] with largest child, and so on...
    max_heapify!(h,1)
    return poppedtophandle,poppedval
end

##=============================================================





##=============================================================
##               MIN stuff
##=============================================================

function build_minheap!(values::Array{Float64,1},Nmax::Integer,
                         handles::Array{Int64,1})
    @assert size(values,1)==size(handles,1)
    ## Init the structure 
    ## Nmax,Nh,nodes,handles
    Nh=size(values,1)
    ## Init big arrays
    ##Array{Float64}(0),10,Array{Int64}(0)
    values2  = zeros(Float64,Nmax)
    handles2 = zeros(Int64,Nmax)
    idxhan2 = zeros(Int64,Nmax)
    
    ## copy to avoid modifying input arrays!!
    values2[1:Nh]  = copy(values)
    handles2[1:Nh] = copy(handles)
    h = BinHeapMin(Nmax,Nh,values2,handles2,idxhan2)

    ## heapify the structure
    sta = div(h.Nh,2)
    for i=sta:-1:1 ## backwards
        min_heapify!(h,i)
    end

    ## get all the addresses of the handles
    for i=1:h.Nh
        h.idxhan[h.handle[i]] = i
    end
   
    return h
end

##=============================================================

function min_heapify!(h::BinHeapMin,i::Int64)
    ## Go DOWN the tree (min heap)...
    # get children indices
    l = index_leftchild(i)
    r = index_rightchild(i)
    ## Introduction to algorithms, p. 154
    if (l <= h.Nh) && (h.nodes[l]<h.nodes[i])
        # smallest on left
        smallest = l
    else
        # smallest on i
        smallest = i
    end
    if (r <= h.Nh) && (h.nodes[r]<h.nodes[smallest])
        # smallest on right
        smallest = r
    end
    if smallest!=i
        # swap the nodes
        swap_nodes_heap!(h,i,smallest)
        # keep going
        min_heapify!(h,smallest)
    end
    return
end


##=============================================================

function update_node_minheap!(h::BinHeapMin,val::Float64,handle::Int64)
    ## find index of node given handle
    ## faster than Julia findfirst ??
    
    # ## print(" find handle, h.Nh = $(h.Nh), time ")
    # idxh = 0
    # @inbounds for l=1:h.Nh
    #     if h.handles[l]==handle
    #         idxh = l
    #         break
    #     end
    # end
    
    idxh = h.idxhan[handle]

    # @show idxh,idxh2
    # @assert idxh==idxh2

    # print(" findfirst-julia  handle, h.Nh = $(h.Nh), time ")
    # @time idxh2 = findfirst(x->x==handle,h.handles)   
    ##@assert size(idxh,1)<=1
    i = idxh #[1] ## from 1D array to scalar
    h.nodes[i] = val
    
    # GO UP THE TREE: compare with parents...
    @inbounds while (i>1) && (h.nodes[index_parent(i)]>h.nodes[i])
        ## swap node with parent
        swap_nodes_heap!(h,i,index_parent(i))
        i = index_parent(i)
    end

    # GO DOWN THE TREE: compare with children...
    if i<h.Nh
        min_heapify!(h,i)
    end
    return
end

##=============================================================

function insert_minheap!(h::BinHeapMin,val::Float64,handle::Int64)
    ## Go UP the tree (min heap) from the bottom...
    # extend heap
    @assert h.Nmax>(h.Nh+1)
    ## IMPORTANT: make sure handle is unique
    ## commented because slows things down
    #   @assert ~(handle in h.handles[1:h.Nh])
    # resize
    h.Nh = h.Nh+1
    # add the new node at the bottom
    h.nodes[h.Nh] = val
    h.handles[h.Nh] = handle # set handle too
    h.idxhan[handle] = h.Nh # set idx of handle too
    
    # start from bottom right leaf
    i = h.Nh
    # compare with parents...
    @inbounds while (i>1) && (h.nodes[index_parent(i)]>h.nodes[i])
        ## swap node with parent
        swap_nodes_heap!(h,i,index_parent(i))
        i = index_parent(i)
    end
    return
end

##=============================================================

function pop_minheap!(h::BinHeapMin)
    # save handle and values of top to return them
    poppedtophandle = h.handles[1]
    poppedval = h.nodes[1]
    # move last elem to top
    h.nodes[1] = h.nodes[h.Nh]
    h.handles[1] = h.handles[h.Nh]
    h.idxhan[h.handles[h.Nh]] = 1 # set idx of handle too    
    ## set out of Nh nodes to 0
    h.nodes[h.Nh] = 0.0
    h.handles[h.Nh] = 0
    ## set to 0 the address of the REMOVED node (poppedtophandle)
    h.idxhan[poppedtophandle] = 0 
    # shorten heap
    h.Nh = h.Nh-1
    # swap the root [1] with largest child, and so on...
    min_heapify!(h,1)
    return poppedtophandle,poppedval
end

##=============================================================

#########################################################
end # end module                                       ##
#########################################################

