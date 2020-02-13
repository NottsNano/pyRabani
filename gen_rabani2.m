% Andrew Stannard 27/06/19, Rabani model in Matlab
clear variables; close all
cm = colormap([0 0 0; 
               1 1 1; 
               1 0.5 0]);
mus = 0:0.2:6;
kTs = 0:0.2:6;

parfor mu_cnt = 1:length(mus)
    for kT_cnt = 1:length(kTs)
        mu = mus(mu_cnt);
        kT = kTs(kT_cnt);
        disp([mu, kT])
        L = 128; N = L^2; % system length and size
        MCS = 450; % total number of Monte Carlo steps
        MR = 30; % mobiity ratio
        C = 0.30; % fractional coverage of nanoparticless
        B = 1/kT; % thermal energy and invrse thermal energy
        e_nl = 1.5; % nanoparticle-liquid interaction energy
        e_nn = 2.0; % nanoparticle-nanoparticle interaction energy
        I = randperm(N,round(C*N)); % random initial nanoparticle positions
        NP = zeros(L); NP(I) = 1; % adding nanoparticles to nanoparticle array
        LQ = ones(L); LQ(I) = 0;  % removing corresponding liquid from liquid array

        % Random number arrays for the diffusion loops
        X1 = ceil(L*rand(MCS, N)); Y1 = ceil(L*rand(MCS, N));
        X2 = ceil(L*rand(MCS, N*MR)); Y2 = ceil(L*rand(MCS, N*MR));
      
        % Nearest and next nearest neighbour arrays with periodic boundaries
        XP11 = X1+1-L*(X1==L); 
        XP21 = X1+2-L*(X1>=L-1);
        YP11 = Y1+1-L*(Y1==L); 
        YP21 = Y1+2-L*(Y1>=L-1);
        XM11 = X1-1+L*(X1==1);         
        XM21 = X1-2+L*(X1<=2);
        YM11 = Y1-1+L*(Y1==1); 
        YM21 = Y1-2+L*(Y1<=2);
        
        XP12 = X2+1-L*(X2==L); 
        XP22 = X2+2-L*(X2>=L-1);
        YP12 = Y2+1-L*(Y2==L); 
        YP22 = Y2+2-L*(Y2>=L-1);
        XM12 = X2-1+L*(X2==1); 
        XM22 = X2-2+L*(X2<=2);
        YM12 = Y2-1+L*(Y2==1); 
        YM22 = Y2-2+L*(Y2<=2);

        R1 = rand(MCS, N,1); % random number array for Metropolis acceptance
        R2 = rand(MCS, N*MR,1); % random number array for Metropolis acceptance

        for m = 1 : MCS % main Monte Carlo loop
            for i = 1 : N % start of evaporation/condensation loop       
                if NP(X1(m, i),Y1(m, i)) == 0     
                    if LQ(X1(m, i),Y1(m, i)) == 0

                        % change in energy if condensation occurs
                        dE = -(LQ(XP11(m, i),Y1(m, i))+LQ(XM11(m, i),Y1(m, i))+LQ(X1(m, i),YP11(m, i))+LQ(X1(m, i),YM11(m, i)))...
                            -e_nl*(NP(XP11(m, i),Y1(m, i))+NP(XM11(m, i),Y1(m, i))+NP(X1(m, i),YP11(m, i))+NP(X1(m, i),YM11(m, i))) + mu;

                        if R1(m, i) < exp(-B*dE) % Metropolis acceptance
                            LQ(X1(m, i),Y1(m, i)) = 1; % condensation
                        end

                    else % i.e. if LQ(x,Y2(m, i)) == 1

                        % change in energy if evaporation occurs
                        dE = (LQ(XP11(m, i),Y1(m, i))+LQ(XM11(m, i),Y1(m, i))+LQ(X1(m, i),YP11(m, i))+LQ(X1(m, i),YM11(m, i)))...
                            +e_nl*(NP(XP11(m, i),Y1(m, i))+NP(XM11(m, i),Y1(m, i))+NP(X1(m, i),YP11(m, i))+NP(X1(m, i),YM11(m, i))) - mu;    
                        if R1(m, i) < exp(-B*dE) % Metropolis acceptance
                            LQ(X1(m, i),Y1(m, i)) = 0; % evaporation
                        end              
                    end
                end        
            end % end of evaporation/condensation loop

            % random number array for nanoparticle movement direction
            D = ceil(4*rand(N*MR,1)); % 1 = left, 2 = right, 3 = down, 4 = up

            for i = 1 : N * MR % start of nanoparticle diffusion loop   
                if NP(X2(m, i),Y2(m, i)) == 1            
                    if (D(i) == 1) && (LQ(XM12(m, i),Y2(m, i)) == 1)

                        % change in energy if nanoparticle moves left
                        dE = -(LQ(X2(m, i),YM12(m, i)) + LQ(XP12(m, i),Y2(m, i)) + LQ(X2(m, i),YP12(m, i)) ...
                            - LQ(XM22(m, i),Y2(m, i)) - LQ(XM12(m, i),YP12(m, i)) - LQ(XM12(m, i),YM12(m, i))) ...
                            -e_nl * (LQ(XM22(m, i),Y2(m, i)) + LQ(XM12(m, i),YP12(m, i)) + LQ(XM12(m, i),YM12(m, i)) ...
                            - LQ(X2(m, i),YM12(m, i)) - LQ(XP12(m, i),Y2(m, i)) - LQ(X2(m, i),YP12(m, i))) ...
                            -e_nl * (NP(X2(m, i),YM12(m, i)) + NP(XP12(m, i),Y2(m, i)) + NP(X2(m, i),YP12(m, i)) ...
                            - NP(XM22(m, i),Y2(m, i)) - NP(XM12(m, i),YP12(m, i)) - NP(XM12(m, i),YM12(m, i))) ...
                            -e_nn * (NP(XM22(m, i),Y2(m, i)) + NP(XM12(m, i),YP12(m, i)) + NP(XM12(m, i),YM12(m, i)) ...
                            - NP(X2(m, i),YM12(m, i)) - NP(XP12(m, i),Y2(m, i)) - NP(X2(m, i),YP12(m, i)));

                        if R2(m, i) < exp(-B*dE) % Metropolis acceptance
                            NP(XM12(m, i),Y2(m, i)) = 1; NP(X2(m, i),Y2(m, i)) = 0; % nanoparticle moves left
                            LQ(X2(m, i),Y2(m, i)) = 1; LQ(XM12(m, i),Y2(m, i)) = 0; % liquid moves right
                        end

                    elseif (D(i) == 2) && (LQ(XP12(m, i),Y2(m, i)) == 1)

                        % change in energy if nanoparticle moves right
                        dE = -(LQ(X2(m, i),YM12(m, i)) + LQ(XP12(m, i),Y2(m, i)) + LQ(X2(m, i),YP12(m, i)) ...
                            - LQ(XP22(m, i),Y2(m, i)) - LQ(XP12(m, i),YP12(m, i)) - LQ(XP12(m, i),YM12(m, i))) ...
                            -e_nl * (LQ(XP22(m, i),Y2(m, i)) + LQ(XP12(m, i),YP12(m, i)) + LQ(XP12(m, i),YM12(m, i)) ...
                            - LQ(X2(m, i),YM12(m, i)) - LQ(XM12(m, i),Y2(m, i)) - LQ(X2(m, i),YP12(m, i))) ...
                            -e_nl * (NP(X2(m, i),YM12(m, i)) + NP(XM12(m, i),Y2(m, i)) + NP(X2(m, i),YP12(m, i)) ...
                            - NP(XP22(m, i),Y2(m, i))-NP(XP12(m, i),YP12(m, i))-NP(XP12(m, i),YM12(m, i))) ...
                            -e_nn * (NP(XP22(m, i),Y2(m, i)) + NP(XP12(m, i),YP12(m, i)) + NP(XP12(m, i),YM12(m, i)) ...
                            - NP(X2(m, i),YM12(m, i)) - NP(XM12(m, i),Y2(m, i)) - NP(X2(m, i),YP12(m, i)));

                        if R2(m, i) < exp(-B*dE) % Metropolis acceptance
                            NP(XP12(m, i),Y2(m, i)) = 1; NP(X2(m, i),Y2(m, i)) = 0; % nanoparticle moves right
                            LQ(X2(m, i),Y2(m, i)) = 1; LQ(XP12(m, i),Y2(m, i)) = 0; % liquid moves left
                        end

                    elseif (D(i) == 3) && (LQ(X2(m, i),YM12(m, i)) == 1)

                        % change in energy if nanoparticle moves down
                        dE = -(LQ(XM12(m, i),Y2(m, i)) + LQ(X2(m, i),YP12(m, i)) + LQ(XP12(m, i),Y2(m, i)) ...
                            - LQ(XM12(m, i),YM12(m, i)) - LQ(X2(m, i),YM22(m, i)) - LQ(XP12(m, i),YM12(m, i))) ...
                            -e_nl * (LQ(XM12(m, i),YM12(m, i)) + LQ(X2(m, i),YM22(m, i)) + LQ(XP12(m, i),YM12(m, i)) ...
                            - LQ(XM12(m, i),Y2(m, i)) - LQ(X2(m, i),YP12(m, i)) - LQ(XP12(m, i),Y2(m, i))) ...
                            -e_nl * (NP(XM12(m, i),Y2(m, i)) + NP(X2(m, i),YP12(m, i)) + NP(XP12(m, i),Y2(m, i)) ...
                            - NP(XM12(m, i),YM12(m, i)) - NP(X2(m, i),YM22(m, i)) - NP(XP12(m, i),YM12(m, i))) ...
                            -e_nn * (NP(XM12(m, i),YM12(m, i)) + NP(X2(m, i),YM22(m, i)) + NP(XP12(m, i),YM12(m, i)) ...
                            - NP(XM12(m, i),Y2(m, i)) - NP(X2(m, i),YP12(m, i)) - NP(XP12(m, i),Y2(m, i)));

                        if R2(m, i) < exp(-B*dE) % Metropolis acceptance
                            NP(X2(m, i),YM12(m, i)) = 1; NP(X2(m, i),Y2(m, i)) = 0; % nanoparticle moves down
                            LQ(X2(m, i),Y2(m, i)) = 1; LQ(X2(m, i),YM12(m, i)) = 0; % liquid moves up 
                        end

                    elseif (D(i) == 4) && (LQ(X2(m, i),YP12(m, i)) == 1)

                        % change in energy if nanoparticle moves up
                        dE = -(LQ(XM12(m, i),Y2(m, i)) + LQ(X2(m, i),YM12(m, i)) + LQ(XP12(m, i),Y2(m, i)) ...
                            - LQ(XM12(m, i),YP12(m, i)) - LQ(X2(m, i),YP22(m, i)) - LQ(XP12(m, i),YP12(m, i))) ...
                            -e_nl * (LQ(XM12(m, i),YP12(m, i)) + LQ(X2(m, i),YP22(m, i)) + LQ(XP12(m, i),YP12(m, i)) ...
                            - LQ(XM12(m, i),Y2(m, i)) - LQ(X2(m, i),YM12(m, i)) - LQ(XP12(m, i),Y2(m, i))) ...
                            -e_nl * (NP(XM12(m, i),Y2(m, i)) + NP(X2(m, i),YM12(m, i)) + NP(XP12(m, i),Y2(m, i)) ...
                            - NP(XM12(m, i),YP12(m, i)) - NP(X2(m, i),YP22(m, i)) - NP(XP12(m, i),YP12(m, i))) ...
                            -e_nn * (NP(XM12(m, i),YP12(m, i)) + NP(X2(m, i),YP22(m, i)) + NP(XP12(m, i),YP12(m, i)) ...
                            - NP(XM12(m, i),Y2(m, i)) - NP(X2(m, i),YM12(m, i)) - NP(XP12(m, i),Y2(m, i)));

                        if R2(m, i) < exp(-B*dE) % Metropolis acceptance
                            NP(X2(m, i),YP12(m, i)) = 1; NP(X2(m, i),Y2(m, i)) = 0; % nanoparticle moves up
                            LQ(X2(m, i),Y2(m, i)) = 1; LQ(X2(m, i),YP12(m, i)) = 0; % liquid moves down 
                        end
                    end
                end
            end
            
        end
        fname = "ImagesMatlabNew2/rabani_kT=" + num2str(kT, '%0.2f') + "mu="  + num2str(mu, '%0.2f') + ".png";
        
        img = (2*NP+LQ)+1;
        imwrite(img, cm, fname);
        
    end
end
