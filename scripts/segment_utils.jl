
abstract type BxB_A end
abstract type B_A   end 
abstract type B_AL2 end

"""
    This filter highlights structures in the input data by comparing local patches around each pixel at different scales and steps. 

    For example, for any pixel at position "pos"...
    ... we will compare the intensity patterns within a 7x7 grid of pixels around "pos"... 
    ... to the intensity patterns in zoomed-out 7x7 grid around "pos"...
    ... the zoomed-out grid is sampled from a smoothed version of the input image with step size of 2 (for example). 

    Since the main idea is to compare patches at different scales/steps, this function accepts pairs of scales/steps as arguments.
    For instance, the code below results in the following 2 local patch comparisons around each pixel: 

    1-. ( 7x7 grid at scale 2 step 2 ) VS ( 7x7 grid at scale 4 step 4 )
    2-. ( 5x5 grid at scale 4 step 4 ) VS ( 5x5 grid at scale 8 step 6 )

    output = multiscale_multistep_filter( input... )
                                          grid_sizes = ( (7,7), (5,5) ),
                                          scales     = ( (2,4), (4,8) ),
                                          steps      = ( (2,4), (4,6) )
                                        )

    # NOTE: this filter is capable of dealing with anistropic resolutions by accepting tuples of values for each "grid_size", "scale"
    and "step"... The code below leads to the followign 2 comparisons:

    1-. ( 7x5 grid at scale (2,1) step (2,1) ) VS ( 7x5 grid at scale (4,2) step (4,2) )
    2-. ( 5x8 grid at scale (4,6) step (4,6) ) VS ( 5x8 grid at scale (8,8) step (6,6) )

    output = multiscale_multistep_filter( input... )
                                          grid_sizes = ( ((7,5),(7,5)), ((5,8),(5,8)) ),
                                          scales     = ( ((2,1),(4,2)), ((4,6),(8,8)) ),
                                          steps      = ( ((2,1),(4,2)), ((4,6),(6,6)) )
                                        )
"""
function multiscale_multistep_filter( inp::Array{<:Real,N}; 
                                      grid_sizes = ( (8,8,2), (8,8,2) ), # IM
                                      scales     = ( ((2,2,1),(3,3,1)), ((3,3,1),(4,4,2) ) ), # rads
                                      steps      = ( ((2,2,1),(3,3,1)), ((3,3,1),(4,4,2) ) ), # steps
                                      dim_downscale=Tuple(ones(Int,N)),
                                      compensate_borders = true,
                                      typ=Float32, 
                                      filter_type=BxB_A, 
                                    ) where {N}
    
    @assert length(scales) == length(steps) == length(grid_sizes)
    num_pairs = length(scales)

    # Filter output
    out = zeros( typ, size( inp ) );

    # Number of operations at each pixel, which is used for averaging the computed quantities
    Narr = zeros( typ, size( inp ) );
 
    # In case a scale is repeated... we only need to compute it once. Just in case., we will 
    # compute the scale space over the unique scales.This requires that we asign to each input
    # scale an index its corresponding scale in the scale-space.

    scales_vector = [ scales[j][i] for i in 1:2, j in 1:length(scales) ][:]
    unique_scales = unique( scales_vector ); 
    scale_indices = [ ( findfirst( x->x==scales[i][1], unique_scales), findfirst( x->x==scales[i][2], unique_scales) )  for i in 1:num_pairs ]

    # Creating the scale space
    scale_space = scale_space_( inp, unique_scales..., compensate_borders=compensate_borders, intA_typ=typ )

    # Squaring the scale-space
    scale_space_2 = scale_space .* scale_space

    # adding the results from the multiscale_step_filter at each pair of scale/steps
    for i in 1:num_pairs;
        scalestep_op!( filter_type, 
                       out, Narr, 
                       scale_space, scale_space_2, 
                       grid_sizes[i], scale_indices[i], steps[i] ); 
    end


    return out ./ max.( 1, Narr )
end

begin ##### SEGMENTATION API

    function multiscale_multistep_segment( inp::Array{<:Real,N}; 
                                           grid_sizes = ( (7,7,5), ),
                                           scales     = ( ((4,4,2),(2,2,1)), ),
                                           steps      = ( ((1,1,1),(4,4,2)), ),
                                           min_dif    = 30, 
                                           alpha      = 1.0,
                                           compensate_borders = true,
                                           typ = Float32, 
                                           filter_type = BxB_A
                                         ) where {N}

        filt = multiscale_multistep_filter( inp, grid_sizes=grid_sizes, scales=scales, steps=steps, typ=typ )
        scsp = scale_space_( inp, scales[1][1], compensate_borders=compensate_borders, intA_typ=typ )
        if N == 2 
            # mask = filt .> ( scsp[:,:,1] .+ min_dif ) .^ alpha

            alpha_ = log.( scsp[:,:,1], min_dif ); 
            mask   =  filt .> ( scsp[:,:,1] ) .^ ( alpha_ .+ alpha )
        else 
            # mask = filt .> ( scsp[:,:,:,1] .+ min_dif ) .^ alpha

            alpha_ = log.( scsp[:,:,:,1], min_dif ); 
            mask   = filt .> ( scsp[:,:,:,1] ) .^ ( alpha_ .+ alpha )
        end
        return mask
    end

end

##### UTILITY FUNCTION 

begin ### MULTISCALE MULTISTEP PATCH FILTER

    # TODO: Replace imaginary by real ffts 
    """
        This function samples two patches at different scales/step around each pixel, and adds
        their "multiscale multistep score" to the output array. 

        See TODO for an illustrative explanation of what is actually computed with this filter.
    """
    function scalestep_op!( ::Type{BxB_A},
                            out::AbstractArray{T,N},             # output array 
                            Narr::AbstractArray{T,N},            # number of operations performed at each pixel, it will be used to compute an average
                            scale_space,                         # scale space of the input data
                            scale_space_2,                       # scale_space squared of the input data
                            grid_size::NTuple{N,Int},            # size of both patches 
                            sidx_pair::NTuple{2,Int},            # scale index for each one of two patches
                            step_pair::NTuple{2,NTuple{N,Int}};  # steps for each one of two patches
                            ) where {T,N}

        # Apart from being sampled at different scales, the two patches are sampled with different
        # "grid steps". The difference between the grid steps determines the kernel that we need
        # to convolve in order to compute the "multiscale multistep score" for each pixel. 

        step_dif    = max.( step_pair[2] .- step_pair[1], 1 );
        grid_kernel = zeros( T, 2 .* grid_size .* step_dif .+ 1 );
        grid_kernel[ Base.StepRange.( 1, step_dif, size(grid_kernel) )... ] .= 1
        
        scale_1_idx = sidx_pair[1]
        scale_2_idx = sidx_pair[2]

        # We want to multiply each pixel in "scale_2" by the values from all surrounding elements 
        # within the "grid_kernel" pattern. This is one half of the "multiscale multistep score" 

        sum_1   = ImageAnalysis.FFTConvolution_crop( scale_space[ axes( out )..., scale_1_idx ], grid_kernel )
        sum_1 .*= view( scale_space, ( axes( out )..., scale_2_idx )... )

        # The second part of the computation involves multiplying each pixel in "scale_2" .^ 2 by the
        # number of surrounding elements within the "grid_kernel" pattern. The pixels in "scale_2" 
        # near the borders require spatial attention... this is what "create_N_grid" is for.

        NN = create_N_grid( grid_size, step_dif, size(out), typ=T ) .- 1; 
        Narr .+= NN

        # All in all, the computation is: scale_2 .^ 2 .* N .- scale_2 .* conv( scale_1, grid_kernel )

        out .+= NN .* view( scale_space_2, ( axes( out )..., scale_2_idx )... ) .- sum_1; 

        return nothing
    end

    function scalestep_op!( ::Type{B_A},
                            out::AbstractArray{T,N},             # output array 
                            Narr::AbstractArray{T,N},            # number of operations performed at each pixel, it will be used to compute an average
                            scale_space,                         # scale space of the input data
                            scale_space_2,                       # scale_space squared of the input data
                            grid_size::NTuple{N,Int},            # size of both patches 
                            sidx_pair::NTuple{2,Int},            # scale index for each one of two patches
                            step_pair::NTuple{2,NTuple{N,Int}};  # steps for each one of two patches
                            ) where {T,N}
        # Kernel
        step_dif    = max.( step_pair[2] .- step_pair[1], 1 );
        grid_kernel = zeros( T, 2 .* grid_size .* step_dif .+ 1 );
        grid_kernel[ Base.StepRange.( 1, step_dif, size(grid_kernel) )... ] .= 1
        
        scale_1_idx = sidx_pair[1]
        scale_2_idx = sidx_pair[2]

        sum_1 = ImageAnalysis.FFTConvolution_crop( scale_space[ axes( out )..., scale_2_idx ], grid_kernel )

        NN = create_N_grid( grid_size, step_dif, size(out) ) .- 1; 

        out .+= view( scale_space, ( axes( out )..., scale_1_idx )... ) .- sum_1 ./ NN

        return nothing
    end

    # (b-a)^2 = b^2 + a^2 - 2*a*b
    function scalestep_op!( ::Type{B_AL2},
                            out::AbstractArray{T,N},             # output array 
                            Narr::AbstractArray{T,N},            # number of operations performed at each pixel, it will be used to compute an average
                            scale_space,                         # scale space of the input data
                            scale_space_2,                       # scale_space squared of the input data
                            grid_size::NTuple{N,Int},            # size of both patches 
                            sidx_pair::NTuple{2,Int},            # scale index for each one of two patches
                            step_pair::NTuple{2,NTuple{N,Int}};  # steps for each one of two patches
                            ) where {T,N}

        # kernel
        step_dif    = max.( step_pair[2] .- step_pair[1], 1 );
        grid_kernel = zeros( T, 2 .* grid_size .* step_dif .+ 1 );
        grid_kernel[ Base.StepRange.( 1, step_dif, size(grid_kernel) )... ] .= 1
        
        scale_1_idx = sidx_pair[1]
        scale_2_idx = sidx_pair[2]

        sum_1   = ImageAnalysis.FFTConvolution_crop( scale_space[ axes( out )..., scale_1_idx ] .* scale_space[ axes( out )..., scale_2_idx ], grid_kernel )

        sum_12  = ImageAnalysis.FFTConvolution_crop( scale_space_2[ axes( out )..., scale_1_idx ], grid_kernel )

        NN = create_N_grid( grid_size, step_dif, size(out) ); 

        out .+= NN .* view( scale_space_2, ( axes( out )..., scale_2_idx )... ) .+ sum_12 .- 2 .* sum_1; 

        return nothing
    end


    begin #### UTILS

        function create_rect_kernel( step1, step2, grid_size; typ=Float32 )
            step_dif    = max.( step2 .- step1, 1 );
            grid_kernel = zeros( typ, 2 .* grid_size .* step_dif .+ 1 );
            grid_kernel[ Base.StepRange.( 1, step_dif, size(grid_kernel) )... ] .= 1
            return grid_kernel
        end

        """
            convolution simulates 0-padding for pixels that are close to the border. This leads to 
            the situation where the output of the filter for pixels around the border incorporates
            less data points than for the other pixels. 
            
            This can be compensated by finding the number of kernel elements for each pixel, which
            can be done with a another convolution... but it can be computed more efficiently from 
            the step and gridsize of the convolved grid.
        """
        function create_N_grid( IM::NTuple{2,Int}, step_dif, inp_size; typ=Float32 )

            output = zeros( typ, inp_size )

            steps  = step_dif .* IM
            c_max  = steps[2]+step_dif[2]
            r_max  = steps[1]+step_dif[1]
            corner = prod( IM .+ 1 )

            for c in 1:div( inp_size[2]+1, 2 ),
                r in 1:div( inp_size[1]+1, 2 )
                
                c_i = min( 2*IM[2]+1, div( c-1, step_dif[2] ) + ( IM[2] + 1 ) )
                r_i = min( 2*IM[1]+1, div( r-1, step_dif[1] ) + ( IM[1] + 1 ) )
                    
                val = c_i * r_i
                
                output[    r   ,    c    ] = val
                output[    r   , end-c+1 ] = val
                output[ end-r+1,    c    ] = val
                output[ end-r+1 ,end-c+1 ] = val
            end

            return output
        end

        function create_N_grid( IM::NTuple{3,Int}, step_dif, inp_size; typ=Float32 )

            output = zeros( typ, inp_size )

            steps  = step_dif .* IM
            z_max  = steps[3]+step_dif[3]
            c_max  = steps[2]+step_dif[2]
            r_max  = steps[1]+step_dif[1]
            corner = prod( IM .+ 1 )

            for z in 1:div( inp_size[3]+1, 2 ), 
                c in 1:div( inp_size[2]+1, 2 ), 
                r in 1:div( inp_size[1]+1, 2 )
                
                z_i = min( 2*IM[3]+1, div( z-1, step_dif[3] ) + ( IM[3] +  1 ) )
                c_i = min( 2*IM[2]+1, div( c-1, step_dif[2] ) + ( IM[2] +  1 ) )
                r_i = min( 2*IM[1]+1, div( r-1, step_dif[1] ) + ( IM[1] +  1 ) )
                
                val = r_i * c_i * z_i 
                
                output[    r   ,    c   ,    z    ] = val
                output[ end-r+1,    c   ,    z    ] = val
                output[    r   , end-c+1,    z    ] = val
                output[    r   ,    c   , end-z+1 ] = val
                output[ end-r+1 ,end-c+1,    z    ] = val
                output[    r   , end-c+1, end-z+1 ] = val
                output[ end-r+1,    c   , end-z+1 ] = val
                output[ end-r+1 ,end-c+1, end-z+1 ] = val
            end

            return output
        end
    end
end

begin ### SCALE-SPACE RELATED FUNCTIONS

    """
        The code below creates a scale space at 3 different scales: (2,2,1), (3,3,1) & (4,4,4). 

        Each scale contains a smoothing radius for each dimension of the input. Each dimensions may
        contain different values, allowing to adapt the smoothing to the resolution of the data. This
        has been included because it is common to have lower resolutions in fluorescence microscopy 
        along the Z dimension, making it to apply less smoothing along the Z axis.s

        scsp = create_scalespace( volume, (2,2,1), (3,3,1), (4,4,4) )
    """
    function scale_space_( input::AbstractArray{T,N}, 
                           unique_scales::Vararg{NTuple{N,Int},ARGC};
                           intA_typ = Float32, 
                           compensate_borders = false
                         ) where {T,N,ARGC}

        # Integral array of the input data, allowing us to compute average intensities around any pixel at arbitrary scales very efficiently.

        intA = ImageAnalysis.integralArray( input, typ=intA_typ );

        # Computing scale space
        num_scales    = length( unique_scales )
        scale_space   = zeros( intA_typ, size( input )..., num_scales ); 

        for i in 1:num_scales
            # Extracting a view to the current scale. Views are references/pointers to a region of an array, and are supposed to avoid unecessary memory copies.
            scale_view = view( scale_space, ( axes( input )..., i )... ) 

            # Computing the smoothed input at the current scale and storing it into the scale_view
            integral_local_avg!( intA, unique_scales[i], scale_view, compensate_borders ); 
        end

        return scale_space
    end

    begin ### INTEGRAL ARRAY UTILITIES

        # with bound checks 

        function integralArea( int_arr::Array{<:AbstractFloat,N}, TL, BR ) where {N}
            @assert all( TL .>= 1 ) && all( BR .< size( int_arr ) ) && all( TL .<= BR )
            return integralArea_unsafe( int_arr, TL, BR )
        end

        # avoiding bound checks... but you will crash julia if you go out-of-bounds

        function integralArea_unsafe( int_arr::Array{<:AbstractFloat,2}, TL, BR )
            TL   = TL .+ 1 .- 1;
            BR   = BR .+ 1;
            @inbounds area = int_arr[BR[1],BR[2]] - int_arr[BR[1],TL[2]] - int_arr[TL[1],BR[2]] + int_arr[TL[1],TL[2]]
            return area
        end

        function integralArea_unsafe( int_arr::Array{<:AbstractFloat,3}, TLF, BRB )
            TLF = TLF .+ 1 .- 1; 
            BRB = BRB .+ 1; 
            @inbounds area  = int_arr[BRB[1],BRB[2],BRB[3]] - int_arr[TLF[1],TLF[2],TLF[3]]
            @inbounds area -= int_arr[TLF[1],BRB[2],BRB[3]] + int_arr[BRB[1],TLF[2],BRB[3]] + int_arr[BRB[1],BRB[2],TLF[3]]
            @inbounds area += int_arr[BRB[1],TLF[2],TLF[3]] + int_arr[TLF[1],BRB[2],TLF[3]] + int_arr[TLF[1],TLF[2],BRB[3]]
            return area
        end

        """
            Computes the local average around the pixel/voxel at position "coord" ...
            ... within the square region of size 2 .+ rads .+ 1 around "coord" ...
            ... by using the integral array "intA" to compute the average intensity ...
            ... and stores the result at position "dest" in the array "out". 
        """
        function integral_local_avg!( intA::Array{T,N},         # integral array of the input
                                      coord::NTuple{N,Int},     # coordinates around which we wish to compute the mean
                                      rads::NTuple{N,Int},      # radii in all dimensions to compute local mean
                                      dest::NTuple{N,Int},      # destination coordinates for each "coords"
                                      out::AbstractArray{Y,N},  # destination array or view to an array
                                      sq_size=prod(2 .*rads.+1) # size of the local squre region, we can pass it as an argument to save computations
                                    ) where {T,Y,N}
            
            # Making sure TLF and BRB are within bounds
            TLF = max.( coord .- rads, 1 );
            BRB = min.( coord .+ rads, size(intA) .- 1 );

            # 
            sq_size = ( sq_size == nothing ) ? prod( BRB .- TLF .+ 1 ) : sq_size;

            # Computing the local sum of intensities with an integral sum
            num = integralArea_unsafe( intA, TLF, BRB );

            # Storing the average intensity in the desired destination
            # TODO: check that "dest" is within bounds of "out"? Or rather leave it unsafe?
            out[ dest... ] = num / sq_size;
            
            return nothing
        end

        """
            Instead of a single "coord" and "dest", this function accepts a list or iterable
            of coordinates and destinations. It iterated over this list and calls the function
            above (integral_local_avg!) for each pair of elements in "coords" and "dests".
            
        """
        function integral_local_avg!( intA, coords, rads, dests, out, compensate_borders::Bool=false; sq_size=prod( 2 .* rads .+  1 ) )
            
            @assert length( coords ) == length( dests ) "coords and dests must contain the same number of elements"    
            
            # reseting the output array
            out .= 0.0

            # computing the local average around each "coord" - "dest" 
            sq_size = ( compensate_borders ) ? nothing : prod( 2 .* rads .+  1 ); 
            for i in 1:length( coords );
                coord = Tuple( coords[i] )
                dest  = Tuple(  dests[i] )
                integral_local_avg!( intA, coord, rads, dest, out, sq_size )
            end
            
            return nothing    
        end

        """
            Simplified version of the previous function for the case when we wish to compute
            the local average of each coordinate in the input array, and we wish to store the
            results in the same order within an array of the same size as the input array.

            In other words, this function assumes that "coord" == "dest".
        """
        function integral_local_avg!( intA, rads, out, compensate_borders::Bool=false; sq_size=prod( 2 .* rads .+  1 ) )
            
            @assert size(intA) == size(out) .+ 1 "dimensions arent compatible"

            # reseting the output array
            out .= 0.0

            # computing the local average around each "coord" - "dest" 
            sq_size = ( compensate_borders ) ? nothing : prod( 2 .* rads .+  1 ); 
            for cartesian_coord in CartesianIndices( axes(out) )
                coord = Tuple( cartesian_coord )
                integral_local_avg!( intA, coord, rads, coord, out, sq_size )
            end

            return nothing    
        end
    end
end

begin ### POST PROCESSING UTILS


    """
        Cleaning the embryo masks of small background noise.
    """
    function clean_mask!( mask; 
                          remove_small = true, min_size=2^3, 
                          remove_borders = true, min_border_distance=(0,0,0),
                          select_dt_maxima = false,
                          mirror_mask = false, zmirror=size(mask,3) 
                        )

        lbls = ImageComponentAnalysis.label_components( mask );

        if ( remove_small || remove_borders ) 
            min_size = min_size.* remove_small
            min_border_distance = min_border_distance .* remove_borders 
            remove_small_and_bordering_CCLs!( lbls, min_vol=min_size, min_border_distance=min_border_distance );
        end
        if select_dt_maxima
            mask .*= distance_transform_maxima( lbls .> 0 )
        else 
            mask .*= lbls .> 0
        end
        if mirror_mask
            mask = mirror_lateral_segmentation( mask, zmirror )
        end
        return mask
    end

    """
        lbls = ImageComponentAnalysis.label_components( seg );
        remove_small_and_bordering_CCLs!( lbls, min_vol=4^3 );  
    """
    function remove_small_and_bordering_CCLs!( lbls::Array{T,N}; 
                                            min_vol=27, 
                                            min_border_distance=zeros(Int,N) ) where {T,N}
        
        # total number of CCLs
        mx_lbl = maximum( lbls );
        N_lbls = mx_lbl + 1; 

        # quantities of interest
        total_elements = zeros( Int, N_lbls ); 
        valid_elements = zeros( Int, N_lbls ); 
        
        h, w, d = size(lbls,1), size(lbls,2), size(lbls,3); 
        min_yxz = (1,1,1) .+ min_border_distance; 
        max_yxz = (h,w,d) .- min_border_distance;
        
        @inbounds for z in 1:d, x in 1:w, y in 1:h
            lbl = lbls[y,x,z]+1
            total_elements[ lbl ] += 1; 
            valid_elements[ lbl ] += all( (y,x,z) .>= min_yxz ) && all( (y,x,z) .<= max_yxz )
        end

        is_big_enough  = total_elements .> min_vol
        isnt_on_border = total_elements .== valid_elements

        for i in 1:length(lbls)
            lbl = lbls[i]+1
            lbls[i] *= is_big_enough[lbl] && isnt_on_border[lbl]
        end
        
        return nothing
    end

    """
    """
    function mirror_lateral_segmentation( mask, zmirror )

        mirrored = copy( mask ); 

        mirrored = zeros( eltype(mask), size(mask)[1:2]..., 2*zmirror )
        mirrored[ :, :,   1:zmirror   ] .=     mask[ :, :,   1:zmirror   ]
        mirrored[ :, :, zmirror+1:end ] .= mirrored[ :, :, zmirror:-1:1  ]

        return mirrored
    end

    """
    """
    function distance_transform_maxima( mask; background_val=0, fmax=8, mx_ovp=(0,0,0), mx_scale=(0,0,0) )

        out = copy( mask ); 
        distance_transform_maxima!( out, background_val=background_val, fmax=fmax, mx_ovp=mx_ovp, mx_scale=mx_scale )
        return out
    end

    function distance_transform_maxima!( mask; background_val=0, fmax=8, mx_ovp=(0,0,0), mx_scale=(0,0,0) )

        f_bool = DistanceTransforms.boolean_indicator( mask .== background_val )
        dt     = DistanceTransforms.transform( f_bool )
        _, mx  = ImageAnalysis.mean_extrema_pad( dt, mx_scale, 0, fmax=fmax, ovp=mx_ovp )
        mask .*= mx; 
        return nothing
    end

end

begin ### MASK UTILS

    # requires ImageDraw, ImageMorphology and ImageComponentAnalysis, QHull
    
    begin # These functions are useful for generating a mask of the embryo for masked_PIV
    
        """
            For 2D masks we can use ImageDraw to fill the edges of the convex hull. From that, 
            we can find the interior of the hull by connected components labelling.  
        """
        function paint_chull_interior( mask::AbstractArray{T,2} ) where {T}
    
            # ch_indices is a Vector{CartesianIndex{2}} with the coordinates of the CH vertices
            ch_indices = ImageMorphology.convexhull( mask ); 
    
            # Creating a Polygon from the ch_indices, which requries a vector of ImageDraw.Points
            ch_points = [ ImageDraw.Point( c[2], c[1] ) for c in ch_indices ]; 
            ch_poly   = ImageDraw.Polygon( ch_points ); 
    
            # Painting convex hull borders... Unfortunately we need to use ColorType colorants...
            ch_mask = zeros( Gray{Float32}, size(mask) )
            ImageDraw.draw!( ch_mask, ch_poly, Gray{Float32}(1) ); 
    
            # We assume that the object of interest is in the center of the image
            ch_centroid = [ 0, 0 ]
            for ci in ch_indices
                ch_centroid .+= Tuple( ci )
            end
            ch_centroid = div.( ch_centroid, length(ch_indices) )
    
            # We compute connected components of the background + interior of the object of interest
            lbls = ImageComponentAnalysis.label_components( ch_mask .== Gray{Float32}(0) ); 
    
            # returning a mask of the pixels with the same label as the center of the mask
            return T.( lbls .== lbls[ ch_centroid... ] )
        end
    
        """
            For 3D masks, we can't use ImageDraw to paint the surface of a 3D convex hull. 
            Instead, I implemented functions to iteratively paint the interior of the faces
            of a 3D convex hull... and any mesh (without intersecting triangles) in general. 
        """
        function paint_chull_interior( mask::AbstractArray{T,3}; val=1 ) where {T}
    
            # Extracting coordinates of true values in mask
            points_mat = mask2point_mat( mask, val=val )
    
            # convex hull with QHull of "points_mat". 
            ch = chull( points_mat ); 
    
            ch_points, ch_faces = qhull2mesh( points_mat, ch ); 
    
            # Painting the surface of the convex hull with DIY ImageDraw.3D
            surface_mask = mesh2mask( ch_points, ch_faces, size(mask); min_size=1)
    
            # dilating the surface to remove small holes
            surface_mask = ImageMorphology.dilate( surface_mask )
            
            # label components of the background + inside of the 3D convex hull
            lbls = ImageComponentAnalysis.label_components( surface_mask .== 0 ); 
            
            # embryo mask = inside of the convex hull
            ch_centroid = mesh_centroid( ch_points ); 
            embryo_mask = UInt8.( lbls .== lbls[ round.( Int, ch_centroid )... ] )
    
            return embryo_mask
        end
    
        function paint_chull_interior( mask_size, mesh_file::String; subsample=subsample, min_size=1 ) where {T}
    
            points, faces, normals = obj2mesh( mesh_file, subsample=subsample )
    
            surface_mask = mesh2mask( points, faces, mask_size, min_size=min_size )
    
            lbls = ImageComponentAnalysis.label_components( surface_mask .== 0 ); 
            
            centroid = mesh_centroid( points ); 
            embryo_mask = UInt8.( lbls .== lbls[ round.( Int, centroid )... ] )
    
            return embryo_mask
        end
    end
    
    
    ###############################################################################################
    
    """
        Given a set of points and faces, create a mask by painting the interior of eachs triangles.
    """
    function mesh2mask( points, faces, mask_size; min_size=1 )
    
        surface_mask = zeros( UInt8, mask_size )
    
        nfaces = size( faces, 1 )
    
        for i in 1:nfaces
            f  = faces[i,:]
            p1 = Tuple( points[ f[1], : ] );
            p2 = Tuple( points[ f[2], : ] );
            p3 = Tuple( points[ f[3], : ] ); 
            iterative_triangle_paint!( surface_mask, p1, p2, p3, min_size )
        end
    
        return surface_mask
    end
    
    """
        Given a set of points and faces, create a mask by painting the interior of eachs triangles.
    """
    function mesh2mask_and_normals( points, faces, normals, mask_size; min_size=1 )
    
        surface_mask  = zeros( UInt8, mask_size )
        normals_array = zeros( eltype( normals ), size(normals,2), mask_size... )
    
        nfaces = size( faces, 1 )
    
        for i in 1:nfaces
            f  = faces[i,:]
            p1 = Tuple(  points[ f[1], : ] );
            p2 = Tuple(  points[ f[2], : ] );
            p3 = Tuple(  points[ f[3], : ] ); 
            n1 = Tuple( normals[ f[1], : ] );
            n2 = Tuple( normals[ f[2], : ] );
            n3 = Tuple( normals[ f[3], : ] ); 
            iterative_triangle_paint!( surface_mask, normals_array, p1, p2, p3, n1, n2, n3, min_size )
        end
    
        return surface_mask, normals_array
    end
    
    begin # Iterative painting of 3D mesh triangles into a mask
    
        function iterative_triangle_paint!( mask::Array{T,N}, p1, p2, p3, Amin ) where {N,T}
            
            p12  = ( p1 .+ p2 ) ./ 2
            p13  = ( p1 .+ p3 ) ./ 2
            p23  = ( p2 .+ p3 ) ./ 2
            p123 = ( p1 .+ p2 .+ p3 ) ./ 3; 
            
            paint_triangle_center!( mask, p123 )
            
            if continue_painting_triangle( mask, p1, p12, p123, Amin )
                iterative_triangle_paint!( mask, p1, p12, p123, Amin )
            end
            if continue_painting_triangle( mask, p1, p13, p123, Amin )
                iterative_triangle_paint!( mask, p1, p13, p123, Amin )
            end
            if continue_painting_triangle( mask, p3, p13, p123, Amin )
                iterative_triangle_paint!( mask, p3, p13, p123, Amin )
            end
            if continue_painting_triangle( mask, p3, p23, p123, Amin )
                iterative_triangle_paint!( mask, p3, p23, p123, Amin )
            end
            if continue_painting_triangle( mask, p2, p23, p123, Amin )
                iterative_triangle_paint!( mask, p2, p23, p123, Amin )
            end
            if continue_painting_triangle( mask, p2, p12, p123, Amin )
                iterative_triangle_paint!( mask, p2, p12, p123, Amin )
            end
            return nothing; 
        end
    
        function iterative_triangle_paint!( mask::Array{T1,N}, normals::Array{T2,M}, 
                                            p1, p2, p3, n1, n2, n3, Amin ) where {T1,T2,N,M}
            
            p12  = ( p1 .+ p2 ) ./ 2
            p13  = ( p1 .+ p3 ) ./ 2
            p23  = ( p2 .+ p3 ) ./ 2
            p123 = ( p1 .+ p2 .+ p3 ) ./ 3; 
    
            n123 = n1 .+ n2 .+ n3; 
            m123 = sqrt.( sum( n123 .* n123 ) ); 
            n123 = ( m123 > 0 ) ? n123 ./ m123 : n123; 
    
            n12 = n1 .+ n2; 
            m12 = sqrt.( sum( n12 .* n12 ) ); 
            n12 = ( m12 > 0 ) ? n12 ./ m12 : n12; 
    
            n13 = n1 .+ n3; 
            m13 = sqrt.( sum( n13 .* n13 ) ); 
            n13 = ( m13 > 0 ) ? n13 ./ m13 : n13; 
    
            n23 = n2 .+ n3; 
            m23 = sqrt.( sum( n23 .* n23 ) ); 
            n23 = ( m23 > 0 ) ? n23 ./ m23 : n23; 
            
            paint_triangle_center!( mask, normals, p123, n123 )
            
            if continue_painting_triangle( mask, p1, p12, p123, Amin )
                iterative_triangle_paint!( mask, normals, p1, p12, p123, n1, n12, n123, Amin )
            end
            if continue_painting_triangle( mask, p1, p13, p123, Amin )
                iterative_triangle_paint!( mask, normals, p1, p13, p123, n1, n13, n123, Amin )
            end
            if continue_painting_triangle( mask, p3, p13, p123, Amin )
                iterative_triangle_paint!( mask, normals, p3, p13, p123, n3, n13, n123, Amin )
            end
            if continue_painting_triangle( mask, p3, p23, p123, Amin )
                iterative_triangle_paint!( mask, normals, p3, p23, p123, n3, n23, n123, Amin )
            end
            if continue_painting_triangle( mask, p2, p23, p123, Amin )
                iterative_triangle_paint!( mask, normals, p2, p23, p123, n2, n23, n123, Amin )
            end
            if continue_painting_triangle( mask, p2, p12, p123, Amin )
                iterative_triangle_paint!( mask, normals, p2, p12, p123, n2, n12, n123, Amin )
            end
            return nothing; 
        end
    
        function paint_triangle_center!( mask::AbstractArray{Bool,N}, 
                                        p123::NTuple{N,Int} ) where {N}
            mask[ p123... ] = true
            return nothing
        end
    
        function paint_triangle_center!( mask::AbstractArray{UInt8,N}, 
                                        p123::NTuple{N,Int} ) where {N}
            mask[ p123... ] = 1
            return nothing
        end
    
        function paint_triangle_center!( mask::AbstractArray{<:Real,N}, 
                                        p123::NTuple{N,T} ) where {N,T<:AbstractFloat}
            
            coords = min.( size(mask), max.( 1, round.( Int, p123 ) ) )
            return paint_triangle_center!( mask, coords ); 
        end
    
        function paint_triangle_center!( mask::AbstractArray{<:Real,N}, 
                                        normals::AbstractArray{T2,M},
                                        p123::NTuple{N,T1}, 
                                        n123::NTuple{N,T2} ) where {N,M,T1<:AbstractFloat,T2<:AbstractFloat}
            
            coords = min.( size(mask), max.( 1, round.( Int, p123 ) ) )
            normals[ :, coords... ] .= n123; 
            return paint_triangle_center!( mask, coords ); 
        end
    
        function continue_painting_triangle( mask, p1, p2, p3, Amin )
            return !is_longside_too_small( p2 .- p1, p3 .- p1, p3 .- p2, Amin ) && !is_triangle_already_painted( mask, p1, p2, p3 )
        end
    
    
        function is_triangle_too_small( p1, p2, p3, Amin )
            return triangle_area( p2 .- p1, p3 .- p2, p1 .- p3 ) < Amin
        end
    
        function is_short_side_too_small( p1, p2, p3, Amin )
            return shortest_side( p2 .- p1, p3 .- p2, p1 .- p3 ) < Amin^2
        end
    
        function is_longside_too_small( p1, p2, p3, Amin )
            return longest_side( p2 .- p1, p3 .- p2, p1 .- p3 ) < Amin^2
        end
    
    
        # https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
        function triangle_area( p12, p13, p23 )
            y, x, z = p12; 
            a, b, c = p13;
            A = 0.5 * sqrt( ( y*c - z*y )^2 + ( z*b - x*z )^2 + ( x*a - y*b )^2 )
            return A
        end
    
        function shortest_side( p12, p13, p23 )
            res1 = min( sum( p12 .^ 2 ), sum( p13 .^ 2 ) )
            res2 = min( res1, sum( p23 .^ 2 ) )
            return res2
        end
    
        function longest_side( p12, p13, p23 )
            res1 = max( sum( p12 .^ 2 ), sum( p13 .^ 2 ) )
            res2 = max( res1, sum( p23 .^ 2 ) )
            return res2
        end
    
    
        function is_triangle_already_painted( mask::AbstractArray{Bool,N}, 
                                                p1::NTuple{N,Int}, 
                                                p2::NTuple{N,Int}, 
                                                p3::NTuple{N,Int} ) where {N}
            
            return false # mask[ p1... ] && mask[ p2... ] && mask[ p3... ]
        end
    
        function is_triangle_already_painted( mask::AbstractArray{UInt8,N}, 
                                                p1::NTuple{N,Int}, 
                                                p2::NTuple{N,Int}, 
                                                p3::NTuple{N,Int} ) where {N}
            
            return false # ( mask[ p1... ] == 1 ) && ( mask[ p2... ] == 1 ) && ( mask[ p3... ] == 1 )
        end
    
        function is_triangle_already_painted( mask::AbstractArray{<:Real,N}, 
                                                p1::NTuple{N,T}, 
                                                p2::NTuple{N,T}, 
                                                p3::NTuple{N,T} ) where {N,T<:AbstractFloat}
            
            return is_triangle_already_painted( mask, round.( Int, p1 ), round.( Int, p2 ), round.( Int, p3 ) ); 
        end
    
        # not true
        function triangle_area_( p12, p13, p23 )
            a = sqrt( sum( p12 .^2 ) )
            b = sqrt( sum( p13 .^2 ) )
            c = sqrt( sum( p23 .^2 ) )
            s = ( a + b + c )/2
            A = sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) )
            return A
        end
    end
    
    begin # Geodesic distance from each pixel of a mask to a user-defined target
    
        function mask_centroid( mask::Array{T,N} ) where {T<:Union{Bool,UInt8},N}
    
            centroid = zeros( Float32, N ); 
            den = 0
            for ci in CartesianIndices( mask ); 
                if Bool( mask[ci] )
                    centroid .+= Tuple( ci )
                    den += 1; 
                end
            end
            return centroid ./ den
        end
    
        function label_stripe( mask, centroid, xz_dir, yx_dir; tol_xz=sind(10), tol_yx=sind(60) )
    
            mag_   = sqrt( sum( xz_dir .^ 2 ) ) 
            xz_dir = xz_dir ./ mag_; 
    
            mag_   = sqrt( sum( yx_dir .^ 2 ) ) 
            yx_dir = yx_dir ./ mag_; 
    
            target = zeros( Bool, size(mask) ); 
            for ci in CartesianIndices( target )
    
                if !Bool( mask[ci] )
                    continue
                end
    
                coord = Tuple( ci );
    
                dir_xz  = coord[2:end] .- centroid[2:end]; 
                mag_xz  = sqrt( sum( dir_xz .^ 2 ) ); 
                if ( mag_xz == 0 )
                    continue
                end
                dirn_xz = dir_xz ./ mag_xz
    
                dir_yx  = coord[1:2] .- centroid[1:2]; 
                mag_yx  = sqrt( sum( dir_yx .^ 2 ) ); 
                if ( mag_yx == 0 )
                    continue
                end
                dirn_yx = dir_yx ./ mag_yx
    
                ndot_xz = sum( dirn_xz .* xz_dir ); 
                ndot_yx = sum( dirn_yx .* yx_dir ); 
                target[ci] = ( ndot_xz >= 1 - tol_xz ) && ( ndot_yx >= 1 - tol_yx )
            end
    
            return target
        end
    
        function geodist( mask_1::Array{T,N}, target::Array{Bool,N} ) where {T,N}
    
            mask = deepcopy( mask_1 )
    
            @assert size( mask ) == size( target )
            inp_size = size( mask ); 
    
            geodists = zeros( Float32, inp_size ) .+ length( mask ); 
            isadded  = zeros( Bool, inp_size ); 
    
            offsets  = [ ( y, x, z ) for y in -1:1, x in -1:1, z in -1:1 ][:]; 
            off_dist = [ sqrt.( sum( off .^ 2 ) ) for off in offsets ]; 
    
            next = []
            for ci in CartesianIndices( target )
                coord = Tuple( ci );
                # for each target pixel
                if target[ci]
                    # add the target's neighbours into the list of seeds if A) the neighbour is in the mask and B) it isn't part of the target
                    for i in 1:length(offsets)
                        offset = offsets[i]
                        dist_  = off_dist[i]
                        neigh  = coord .+ offset
                        if all( neigh .>= 1 ) && all( neigh .<= inp_size )
                            if !target[neigh...] && Bool( mask[neigh...] )
                                if !isadded[neigh...]
                                    push!( next, neigh )
                                end
                                isadded[neigh...] = true
                                mask[neigh...] = 0
                                geodists[neigh...] = min( geodists[neigh...], dist_ )
                            end
                        end
                    end
                else
                    continue
                end
            end
    
            Nmask_remaining = length( next )
            for i in 1:length(mask)
                Nmask_remaining += Bool( mask[i] ) && !target[i]
            end
    
            while Nmask_remaining > 0
    
                new_next = []
                for coord in next
                    mask[ coord... ] = 0
                    Nmask_remaining -= 1
                    for i in 1:length(offsets)
                        offset = offsets[i]
                        dist_  = off_dist[i]
                        neigh  = coord .+ offset
                        if all( neigh .>= 1 ) && all( neigh .<= inp_size )
                            if !target[neigh...] && Bool( mask[neigh...] )
                                if !isadded[ neigh... ]
                                    push!( new_next, neigh )
                                end
                                isadded[ neigh... ] = true
                                geodists[ neigh... ] = min( geodists[ neigh... ], geodists[ coord... ] + dist_ )
                            end
                        end
                    end
                end
                next = new_next;  
            end
    
            geodists .*= mask_1 .* ( target .== false );
            geodists[ geodists .== length( mask ) ] .= 0.0
    
            return geodists, next
        end
    
        function direction_graddescent( gdist; ksize=(5,5,5) )
    
            kern_y = Float32[ y / sqrt( sum( (y,x,z) .^2 ) )  for y in -ksize[1]:ksize[1], x in -ksize[2]:ksize[2], z in -ksize[3]:ksize[3] ]
            kern_x = Float32[ x / sqrt( sum( (y,x,z) .^2 ) )  for y in -ksize[1]:ksize[1], x in -ksize[2]:ksize[2], z in -ksize[3]:ksize[3] ]
            kern_z = Float32[ z / sqrt( sum( (y,x,z) .^2 ) )  for y in -ksize[1]:ksize[1], x in -ksize[2]:ksize[2], z in -ksize[3]:ksize[3] ]
            kern_y[ isnan.( kern_y ) ] .= 0
            kern_x[ isnan.( kern_x ) ] .= 0
            kern_z[ isnan.( kern_z ) ] .= 0
    
            grad_y = multi_quickPIV.FFTCC_crop( kern_y, gdist )
            grad_x = multi_quickPIV.FFTCC_crop( kern_x, gdist )
            grad_z = multi_quickPIV.FFTCC_crop( kern_z, gdist )
    
            mags = sqrt.( grad_y .^ 2 .+ grad_x .^ 2 .+ grad_z .^ 2 );
            mags[ mags .== 0 ] .= 1 
    
            grad_y ./= mags; 
            grad_x ./= mags; 
            grad_z ./= mags; 
            
            grad_y[ gdist .< 1 ] .= 0
            grad_x[ gdist .< 1 ] .= 0
            grad_z[ gdist .< 1 ] .= 0
    
            return grad_y, grad_x, grad_z
        end
    
        function direction_graddescent_2( gdist; ksize=(5,5,5) )
    
            grad_y = zeros( Float32, size( gdist ) )
            grad_x = zeros( Float32, size( gdist ) )
            grad_z = zeros( Float32, size( gdist ) )
    
            h, w, d = size( gdist ); 
    
            for ci in CartesianIndices( gdist )
                if gdist[ci] == 0 
                    continue
                end
                coord = Tuple( ci )
    
                min_gdist = Inf; 
                min_dir = Float32[ 0, 0, 0 ]
                N = 0
                for z in max(1,ci[3]-ksize[3]):min(d,ci[3]+ksize[3]), 
                    x in max(1,ci[2]-ksize[2]):min(w,ci[2]+ksize[2]), 
                    y in max(1,ci[1]-ksize[1]):min(h,ci[1]+ksize[1]) 
    
                    if gdist[y,x,z] == 0
                        continue
                    end
    
                    if gdist[y,x,z] < min_gdist
                        min_gdist = gdist[y,x,z]
                        min_dir  .= (y,x,z) .- coord 
                        N = 1
                    elseif gdist[y,x,z] == min_gdist
                        min_dir  .+= (y,x,z) .- coord 
                        N += 1
                    end
                end
    
                min_dir ./= max( 1, N )
    
                mag_ = sqrt( sum( min_dir .^ 2 ) ); 
                if mag_ > 0
                    min_dir = min_dir ./ mag_
                end
    
                grad_y[ ci ] = min_dir[1]
                grad_x[ ci ] = min_dir[2]
                grad_z[ ci ] = min_dir[3]
            end
    
            return grad_y, grad_x, grad_z
        end
    
        function smooth_gradient( gy, gx, gz; ksize=(5,5,5) )
    
            avg_kern = ones( Float32, 2 .* ksize .+ 1 )
    
            avg_y = multi_quickPIV.FFTCC_crop( avg_kern, gy )
            avg_x = multi_quickPIV.FFTCC_crop( avg_kern, gx )
            avg_z = multi_quickPIV.FFTCC_crop( avg_kern, gz )
    
            mags = sqrt.( avg_y .^ 2 .+ avg_x .^ 2 .+ avg_z .^ 2 );
            mags[ mags .== 0 ] .= 1 
            avg_y ./= mags; 
            avg_x ./= mags; 
            avg_z ./= mags; 
            
            mask = sqrt.( gy .^ 2 .+ gx .^ 2 .+ gz .^ 2 ) .< 1
            avg_y[ mask ] .= 0.0
            avg_x[ mask ] .= 0.0
            avg_z[ mask ] .= 0.0
    
            return avg_y, avg_x, avg_z
        end
    end

end

begin ### MESH UTILS

    # meshes represented as points + faces [ + normals ] arrays. 

    begin # very common operations

        function list2matrix( list; op=(x)->(x))
            nelements = length( list ); 
            ncolumns  = length( list[1] ); 
            T = eltype( list[1] ); 
            matrix = zeros( T, nelements, ncolumns)
            for i in 1:nelements
                matrix[i,:] .= op( list[i] )
            end
            return matrix
        end

        function matrix2list( matrix; op=(x)->(Tuple(x)) )
            nelements = size( matrix, 1 ); 
            list = [ op( matrix[i,:] ) for i in 1:nelements ]
            return list
        end

        function mask2point_mat( mask; val=1 )
            points = CartesianIndices(mask)[mask .== val]; 
            return list2matrix( points, op=(x)->(Tuple(x)) )  
        end

        function mesh_centroid( points )
            centroid  = [ 0.0, 0.0, 0.0 ]
            for i in 1:size(points,1)
                centroid .+= points[ i, : ]
            end
            centroid ./= size(points,1)
            return centroid
        end
    end

    begin # .obj IO

        begin # Basic .obj IO

            """
                Saving mesh to obj file
            """
            function mesh2obj( points, faces, output; scale=(1,1,1) )
                
                npoints = size( points, 1 ); 
                nfaces  = size( faces , 1 );

                lines_obj ="o obj\n";
                for i in 1:npoints
                    lines_obj *= "v " * join( string.( points[i,:] .* scale ), " " ) * "\n"; 
                end
                for i in 1:nfaces
                    lines_obj *= "f " * join( string.( faces[i,:] ), " " ) * "\n"; 
                end
                lines_obj *= "# Vertices: $(npoints), normals: 0, texture coordinates: 0, faces: $(nfaces)"

                write( output, lines_obj )
            end

            function mesh2obj( points, faces, normals, output; scale=(1,1,1) )
                
                npoints = size( points, 1 ); 
                nfaces  = size( faces , 1 );
                nnormals = size( normals, 1 );

                lines_obj ="o obj\n";
                for i in 1:npoints
                    lines_obj *= "v " * join( string.( points[i,:] .* scale ), " " ) * "\n"; 
                end
                for i in 1:nfaces
                    lines_obj *= "f " * join( string.( faces[i,:] ), " " ) * "\n"; 
                end
                for i in 1:nnormals
                    lines_obj *= "vn " * join( string.( normals[i,:] ), " " ) * "\n"; 
                end
                lines_obj *= "# Vertices: $(npoints), normals: $(nnormals), texture coordinates: 0, faces: $(nfaces)"

                write( output, lines_obj )
            end

            """
                Loading .obj mesh
            """
            function obj2mesh( obj_file; subsample=(1,1,1) )
                
                fobj = open( obj_file )
                lines_obj = readlines( fobj )
                close( fobj )
                
                # pushing vertices and faces into vectors of tuples
                points = []
                faces  = []
                normals = []
                for i in 1:length(lines_obj)
                    line = lines_obj[i]
                    if length(line) < 2
                        continue; 
                    end
                    if line[1:2] == "vn"
                        push!( normals, parse.( Float64, split( line, " " )[1+1:3+1] ) )
                    elseif line[1] == 'v'
                        push!( points, parse.( Float64, split( line, " " )[1+1:3+1] ) .* subsample )
                    elseif line[1] == 'f'
                        if occursin( "//", line )
                            split1 = split( line, " " )[1+1:3+1] 
                            split2 = [ split( x, "//" )[1] for x in split1 ]; 
                            push!( faces, parse.( Int, split2 ) )
                        else
                            push!( faces, parse.( Int, split( line, " " )[1+1:3+1] ) )
                        end
                    end
                end

                # converting the vector of tuples into matrices
                
                points_mat  = list2matrix( points  )
                faces_mat   = list2matrix( faces   )
                normals_mat = ( length(normals) > 0 ) ? list2matrix( normals ) : nothing
                
                return points_mat, faces_mat, normals_mat
            end
        end 

        begin # Meshlab .obj format

            """
                ImSAnE expects .obj files generated by meshlab, which have a special format. 
            """
            function mesh2mlbobj( points, faces, normals, output; scale=(1,1,1), flip_xy=false )

                @assert size( points,1 ) == size( normals, 1 )

                npoints  = size( points , 1 ); 
                nfaces   = size( faces  , 1 );
                nnormals = size( normals, 1 );

                header = """####
                #
                # OBJ File copying Meshlab's .obj format
                #
                ####
                # Object flip_xy $(flip_xy)
                #
                # Vertices: $(npoints)
                # Faces: $(nfaces)
                #
                ####\n"""

                lines_obj = header;
                for i in 1:npoints
                    norm  = ( normals[i,1+flip_xy], normals[i,2-flip_xy], normals[i,3] ); 
                    point = ( points[i,1+flip_xy], points[i,2-flip_xy], points[i,3] ) .* ( scale[1+flip_xy], scale[2-flip_xy], scale[3] ) 
                    lines_obj *= "vn " * join( string.( norm  ), " " ) * "\n";
                    lines_obj *= "v "  * join( string.( point ), " " ) * "\n";
                end
                lines_obj *= "# $(npoints) vertices, 0 vertices normals\n\n"

                for i in 1:nfaces
                    face_txt = "f " * string(faces[i,1]) * "//" * string(faces[i,1]) * " " * string(faces[i,2]) * "//" * string(faces[i,2]) * " " * string(faces[i,3]) * "//" * string(faces[i,3]) * "\n"
                    lines_obj *= face_txt; 
                end
                lines_obj *= "# $(nfaces) faces, 0 coords texture\n\n"
                lines_obj *= "# End of File"

                write( output, lines_obj )
            end
        end 

        begin # Converting between .obj and other formats ( meshlab_obj, vtk, etc )

            function obj2mlbobj( obj_file, out_file; flip_xy=false )

                points, faces, normals = obj2mesh( obj_file ); 
                mesh2mlbobj( points, faces, normals, out_file, flip_xy=flip_xy ); 
                return nothing
            end

            function obj2vtk( obj_file, vtp_file )

                points, faces, normals = obj2mesh( obj_file );
                fens = FinEtools.FENodeSet( points ); 
                fes  = FinEtools.FESetT3( faces ); 
                vtkexportmesh( vtp_file, fens, fes); 
                return nothing
            end
        end

        begin # .off IO

            function points2off( points, off_file; typ=Float32, scale=(1,1,1) )

                Npoints = size( points, 1 ); 
                off_txt = "OFF\n$(Npoints) 0 0\n"
                for n in 1:Npoints
                    off_txt *= join( typ.( points[n,:] ) .* scale, " " ) * "\n"
                end
                write( off_file, off_txt )

                return nothing
            end

            function off2obj( off_file; obj_file = replace( off_file, ".off" => ".obj") )
                foff = open( off_file )
                lines_off = readlines( foff )
                nvertices = 0
                nfaces    = 0
                lines_obj = "o obj\n"
                for line in lines_off[4:end-1]
                    if line[1:2] == "3 "
                        nums = parse.( Int, split( line[4:end], " " ) )
                        fline = "f "; 
                        [ fline *= string(num + 1) * " " for num in nums ]; 
                        lines_obj *= fline * "\n"
                        nfaces += 1
                    else
                        lines_obj *= "v " * line * "\n" 
                        nvertices += 1
                    end
                end
                lines_obj *= "# Vertices: $(nvertices), normals: 0, texture coordinates: 0, faces: $(nfaces)"
                write( obj_file, lines_obj )
                close( foff )
            end

        end
    end

    begin # FinEtools to smooth the mesh

        function smooth_mesh( file_in::String, file_out::String; method=:taubin, npass=10, flip_xy=true)

            points, faces, normals = obj2mesh( file_in )
            s_points, s_faces = smooth_mesh( points, faces, method=method, npass=npass )
            s_normals = compute_normals_with_centroid( s_points, s_faces )
            mesh2mlbobj( s_points, s_faces, s_normals, file_out, flip_xy=flip_xy )
            return nothing
        end

        function smooth_mesh( points, faces; method=:taubin, npass=10 )
            
            # converting to FinEtoolsobjects
            fens = FinEtools.FENodeSet( points ); 
            fes  = FinEtools.FESetT3( faces ); 
            
            # smoothing mesh
            fens = meshsmoothing(fens, fes, method=method, npass=npass )
            
            # subdividing mesh
            fens, fes = T3refine( fens, fes ); 
            
            smooth_points = fens.xyz; 
            smooth_faces  = list2matrix( fes.conn )
            
            return smooth_points, smooth_faces
        end
    end

    begin # Qhull utils

        """
            Computes the convex hull of a 3D mask. 
        """
        function mask2chull( mask::AbstractArray{T,3}; val=1 ) where {T}

            # Extracting coordinates of true values in mask
            points_mat = mask2point_mat( mask, val=val )

            # convex hull with QHull of "points_mat". 
            ch = chull( points_mat );         

            return ch
        end

        function mask2chull_and_normals( mask::AbstractArray{T,3}; val=1 ) where {T}

            # Extracting coordinates of true values in mask
            points_mat = mask2point_mat( mask, val=val )

            # convex hull with QHull of "points_mat". 
            ch = chull( points_mat ); 
            
            # computing normals with the cross product and orienting them to point away from the centroid
            ch_normals = compute_normals_with_centroid( points_mat, ch )

            return ch, ch_normals
        end

        begin # functions to compute normals 

            function compute_normals_with_centroid( points::Matrix{T}, ch ) where {T}

                ch_points, ch_faces = qhull2mesh( points, ch )
                ch_normals = compute_normals_with_centroid( ch_points, ch_faces )
                return ch_normals
            end

            function compute_normals_with_centroid( points::Matrix{T1}, faces::Matrix{T2} ) where {T1,T2}

                centroid = mesh_centroid( points )

                nfaces  = size( faces, 1 )
                npoints = size( points, 1 ); 
                normals = zeros( eltype( points ), npoints, 3 )

                for n in 1:nfaces
                    face = faces[n,:]
                    vn = outward_normal( points, face, centroid )
                    normals[ face[1], : ] .+= vn
                    normals[ face[2], : ] .+= vn
                    normals[ face[3], : ] .+= vn
                end

                for n in 1:npoints
                    vn  = normals[ n, : ]
                    mag = sqrt( sum( vn .^ 2 ) )
                    normals[ n, : ] ./= mag
                end

                return normals
            end

            function outward_normal( points, face, centroid )

                p1, p2, p3 = points[face[1],:], points[face[2],:], points[face[3],:]; 
                normal     = cross_( p2 .- p1, p3 .- p1 );
                normal   ./= mag_( normal )
                vec2center = centroid .- p1;
                return -1 * sign(dot_(vec2center,normal)) .* normal;
            end

            mag_( a ) = sqrt( dot_( a, a ) ); 
            dot_( a, b ) = dot_muladd( a, b )
            cross_( a, b ) = cross_muladd( a, b ) 
            # dot and cross products of using muladd... which is supposed to compute a multiplication and addition simulatenouslys
            dot_muladd( a, b )  = muladd( a[1], b[1], a[2]*b[2] ) + a[3]*b[3];
            cross_muladd( a, b ) = [ muladd( a[2], b[3], -1*a[3]*b[2] ), muladd( a[3], b[1], -1*a[1]*b[3] ), muladd( a[1], b[2], -1*a[2]*b[1] ) ]
        end

        """
            Computes the convex hull of a 3D mask, and stores the 
            results as an .obj file
        """
        function mask2chull2obj( mask, obj_file; scale=(1,1,1), val=1 )

            # Extracting coordinates of true values in mask
            points_mat = mask2point_mat( mask, val=val )

            # convex hull with QHull of "points_mat". 
            ch = chull( points_mat ); 

            # extracting the set of points and faces from ch
            ch_points, ch_faces = qhull2mesh( points_mat, ch ); 

            ch_normals = compute_normals_with_centroid( points_mat, ch )

            mesh2obj( ch_points, ch_faces, ch_normals, obj_file, scale=scale )

            return nothing;
        end

        """
            ch.vertices and ch.faces are indices to the input points, so we need to
            constantly carry the input points around. 

            This functions samples the set of points of ch.vertices, and creates a
            new ch.faces, whose indices refer to the samples points[ ch.vertices ]
        """
        function qhull2mesh( points, qhull )

            nverts = length( qhull.vertices ); 
            nfaces = size( qhull.simplices, 1 ); 
            point2vertices = zeros( Int, length(points) );

            hull_points = zeros( Float64, nverts, 3 )
            for i in 1:nverts
                hull_points[i,:] .= points[qhull.vertices[i],:]
                point2vertices[qhull.vertices[i]] = i
            end

            hull_faces = zeros( Int, nfaces, 3 ); 
            for i in 1:nfaces
                f1, f2, f3 = qhull.simplices[i,:]
                hull_faces[i,1] = point2vertices[f1]
                hull_faces[i,2] = point2vertices[f2]
                hull_faces[i,3] = point2vertices[f3]
            end

            return hull_points, hull_faces
        end

    end

    begin # Meshlab utils


        function apply_mlxfilter( obj_file, mlx_file, out_file )

            pymeshlab = pyimport("pymeshlab")
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh( obj_file )
            ms.load_filter_script( mlx_file )
            ms.apply_filter_script()
            ms.save_current_mesh( out_file )
            return nothing;
        end

        function install_pymeshlab()

            env = Conda.ROOTENV
            Conda.pip_interop( true, env ); 
            Conda.pip( "install --user", "pymeshlab", env )
            println("Now you need to restart the notebook to be able to import the package with PyCall"); 
            return nothing;
        end

    end

    begin # Rotate meshes

        function rotate_mesh_xy( angle, file_in::String, file_out::String )

            points, faces, normals = obj2mesh( file_in )

            rot_points, rot_normals = rotate_mesh( angle, points, normals, fun=rotate_xy )

            rot_points[1,:] .+= 

            mesh2mlbobj( rot_points, faces, rot_normals, file_out )
        end

        function rotate_mesh_xz( angle, file_in::String, file_out::String )

            points, faces, normals = obj2mesh( file_in )

            rot_points, rot_normals = rotate_mesh( angle, points, normals, fun=rotate_xz )

            mesh2mlbobj( rot_points, faces, rot_normals, file_out )
        end

        function rotate_mesh( angle, points, normals; fun=rotate_xy )

            rot_points  = copy( points ); 
            rot_normals = copy( normals ); 

            for i in 1:size( points, 1 ) 
                rot_points[i,:]  .= fun( points[i,:], angle )
                rot_normals[i,:] .= fun( normals[i,:], angle )
            end

            return rot_points, rot_normals
        end


        function rotate_xy( point, angle )
            rot_x = point[1] * cosd( angle ) - point[2] * sind( angle )
            rot_y = point[1] * sind( angle ) + point[2] * cosd( angle )
            return rot_x, rot_y, point[3]
        end

        function rotate_xz( point, angle )
            rot_x = point[1] * cosd( angle ) - point[3] * sind( angle )
            rot_z = point[1] * sind( angle ) + point[3] * cosd( angle )
            return rot_x, point[2], rot_z
        end
    end


end