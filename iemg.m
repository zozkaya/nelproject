function [iemg_data,t_new] = iemg(data,t)

    
    bin_length = .005; % select bin length over time 
    shift = .005;
    first = 0;
    last = 0; 

    iemg_data = []; 
    t_new = [] ;
    
    while first < t(end) - bin_length 
        last = first + bin_length; 
    
        ind_1 = find(t>=first,1);
        ind_2 = find(t>=last,1); 
        
        bin = data(ind_1:ind_2); 
        val = sum(abs(bin));
        
        t_new = [t_new; first]; 
        iemg_data = [iemg_data; val];
        first = first + shift;
    end 
    
    


end 