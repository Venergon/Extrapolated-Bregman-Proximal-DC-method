function x = perform_cg(A,y,options)

// perform_cg - conjugate gradient
//  
//  x = perform_cg(A,y,options);
//
//  Copyright (c) 2009 Gabriel Peyre

options.null = 0;
tol = getoptions(options, 'tol', 1d-6);
maxit = getoptions(options, 'maxit', 100);

[x, fail, err, iter, res] = pcg(A,y,maxIter=maxit,tol=tol);

endfunction