function [fmd_data,t_new] = fmd(data,t) %need to check this code 

    
    bin_length = .005;
    shift = .005;
    first = 0;
    last = 0; 

    fmd_data = []; 
    t_new = [] ;
    
    while first < t(end) - bin_length 
        last = first + bin_length; 
    
        ind_1 = find(t>=first,1);
        ind_2 = find(t>=last,1); 
        
        bin = data(ind_1:ind_2); 
        val = .5*sum(periodogram(bin));
        
        t_new = [t_new; first]; 
        fmd_data = [fmd_data; val];
        first = first + shift;
    end 
    
    


end 
