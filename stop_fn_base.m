function [stop] = stop_fn_base(obj_fn, rtol, x_hat, x_prev, x_curr, iteration) 
    obj_difference = obj_fn(x_prev) - obj_fn(x_curr);
    
    stop = 0;
    
     if (mod(iteration, 1000) == 0)
        fprintf('Iteration: %d\n', iteration);
        fprintf('Previous 2 obj values: %e %e\n', obj_fn(x_prev), obj_fn(x_curr));
        fprintf('Diff: %e\n', obj_fn(x_prev) - obj_fn(x_curr));
     end
    
    if (obj_difference < 0) 
        fprintf('Error: iteration: %d obj_difference %e is negative\n', iteration, obj_difference);
        %fprintf('Prev x %e, curr x %e diff %e\n', norm(x_prev, 2), norm(x_curr, 2), norm(x_prev - x_curr, 2));
        %throw(MException('TEST'));
    elseif (obj_difference < rtol*obj_fn(x_hat))
        stop = 1;
    end
    
end
