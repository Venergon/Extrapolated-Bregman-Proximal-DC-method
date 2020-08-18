function A = make_sparse(i,j,x)

A = sparse( [i(:) j(:)], x(:) );

endfunction