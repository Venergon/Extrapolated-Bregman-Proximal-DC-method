function x = perform_cg(A,y,options)

options.null = 0;
tol = getoptions(options, 'tol', 1e-6);
maxit = getoptions(options, 'maxit', 100);

[x,flag] = cgs(A,y,tol,maxit);