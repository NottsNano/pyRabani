% Andrew Stannard 27/06/19, Rabani model in Matlab
clear variables; 

mus = [0:1:6];
kTs = [0:0.2:1];
        cm = colormap([0 0 0; 1 1 1; 1 0.5 0]);
parfor mu_cnt = 1:length(mus)
    for kT_cnt = 1:length(kTs)
        disp([mu_cnt, kT_cnt])
        L = 1024; 
        N = L^2; % system length and size
        MCS = 1000; % total number of Monte Carlo steps
        MR = 30; % mobiity ratio
        C = 0.25; % fractional coverage of nanoparticles
        kT = kTs(kT_cnt); B = 1/kT; % thermal energy and invrse thermal energy
        e_nl = 1.5; % nanoparticle-liquid interaction energy
        e_nn = 2.0; % nanoparticle-nanoparticle interaction energy
        mu = mus(mu_cnt); % liquid chemical potential
        % substrate = black, liquid = white, nanoparticle = orange


        I = randperm(N,round(C*N)); % random initial nanoparticle positions
        NP = zeros(L); NP(I) = 1; % adding nanoparticles to nanoparticle array
        LQ = ones(L); LQ(I) = 0;  % removing corresponding liquid from liquid array

        for m = 1 : MCS % main Monte Carlo loop

            % random position arrays for the evaporation/condensation loop
            X = ceil(L*rand(N,1)); Y = ceil(L*rand(N,1));

            % nearest and next nearest neighbour arrays with periodic boundaries
            XP1 = X+1-L*(X==L); 
            XP2 = X+2-L*(X>=L-1);
            YP1 = Y+1-L*(Y==L); 
            YP2 = Y+2-L*(Y>=L-1);
            XM1 = X-1+L*(X==1);         
            XM2 = X-2+L*(X<=2);
            YM1 = Y-1+L*(Y==1); 
            YM2 = Y-2+L*(Y<=2);
            R = rand(N,1); % random number array for Metropolis acceptance

            for i = 1 : N % start of evaporation/condensation loop

                x = X(i); xp1 = XP1(i); xp2 = XP2(i); xm1 = XM1(i); xm2 = XM2(i);
                y = Y(i); yp1 = YP1(i); yp2 = YP2(i); ym1 = YM1(i); ym2 = YM2(i);
                r = R(i);

                if NP(x,y) == 0

                    if LQ(x,y) == 0

                        % change in energy if condensation occurs
                        dE = -(LQ(xp1,y)+LQ(xm1,y)+LQ(x,yp1)+LQ(x,ym1))...
                            -e_nl*(NP(xp1,y)+NP(xm1,y)+NP(x,yp1)+NP(x,ym1)) + mu;

                        if r < exp(-B*dE) % Metropolis acceptance
                            LQ(x,y) = 1; % condensation
                        end

                    else % i.e. if LQ(x,y) == 1

                        % change in energy if evaporation occurs
                        dE = (LQ(xp1,y)+LQ(xm1,y)+LQ(x,yp1)+LQ(x,ym1))...
                            +e_nl*(NP(xp1,y)+NP(xm1,y)+NP(x,yp1)+NP(x,ym1)) - mu;

                        if r < exp(-B*dE) % Metropolis acceptance
                            LQ(x,y) = 0; % evaporation
                        end

                    end

                end

            end % end of evaporation/condensation loop

            % random number arrays for the nanoparticle diffusion loop
            X = ceil(L*rand(N*MR,1)); Y = ceil(L*rand(N*MR,1));
            % nearest and next nearest neighbour arrays with periodic boundaries
            XP1 = X+1-L*(X==L); XP2 = X+2-L*(X>=L-1);
            YP1 = Y+1-L*(Y==L); YP2 = Y+2-L*(Y>=L-1);
            XM1 = X-1+L*(X==1); XM2 = X-2+L*(X<=2);
            YM1 = Y-1+L*(Y==1); YM2 = Y-2+L*(Y<=2);
            R = rand(N*MR,1); % random number array for Metropolis acceptance
            % random number array for nanoparticle movement direction
            D = ceil(4*rand(N*MR,1)); % 1 = left, 2 = right, 3 = down, 4 = up

            for i = 1 : N * MR % start of nanoparticle diffusion loop

                x = X(i); xp1 = XP1(i); xp2 = XP2(i); xm1 = XM1(i); xm2 = XM2(i);
                y = Y(i); yp1 = YP1(i); yp2 = YP2(i); ym1 = YM1(i); ym2 = YM2(i);
                r = R(i); d = D(i);

                if NP(x,y) == 1

                    if (d == 1) && (LQ(xm1,y) == 1)

                        % change in energy if nanoparticle moves left
                        dE = -(LQ(x,ym1) + LQ(xp1,y) + LQ(x,yp1) ...
                            - LQ(xm2,y) - LQ(xm1,yp1) - LQ(xm1,ym1)) ...
                            -e_nl * (LQ(xm2,y) + LQ(xm1,yp1) + LQ(xm1,ym1) ...
                            - LQ(x,ym1) - LQ(xp1,y) - LQ(x,yp1)) ...
                            -e_nl * (NP(x,ym1) + NP(xp1,y) + NP(x,yp1) ...
                            - NP(xm2,y) - NP(xm1,yp1) - NP(xm1,ym1)) ...
                            -e_nn * (NP(xm2,y) + NP(xm1,yp1) + NP(xm1,ym1) ...
                            - NP(x,ym1) - NP(xp1,y) - NP(x,yp1));

                        if r < exp(-B*dE) % Metropolis acceptance
                            NP(xm1,y) = 1; NP(x,y) = 0; % nanoparticle moves left
                            LQ(x,y) = 1; LQ(xm1,y) = 0; % liquid moves right
                        end

                    elseif (d == 2) && (LQ(xp1,y) == 1)

                        % change in energy if nanoparticle moves right
                        dE = -(LQ(x,ym1) + LQ(xp1,y) + LQ(x,yp1) ...
                            - LQ(xp2,y) - LQ(xp1,yp1) - LQ(xp1,ym1)) ...
                            -e_nl * (LQ(xp2,y) + LQ(xp1,yp1) + LQ(xp1,ym1) ...
                            - LQ(x,ym1) - LQ(xm1,y) - LQ(x,yp1)) ...
                            -e_nl * (NP(x,ym1) + NP(xm1,y) + NP(x,yp1) ...
                            - NP(xp2,y)-NP(xp1,yp1)-NP(xp1,ym1)) ...
                            -e_nn * (NP(xp2,y) + NP(xp1,yp1) + NP(xp1,ym1) ...
                            - NP(x,ym1) - NP(xm1,y) - NP(x,yp1));

                        if r < exp(-B*dE) % Metropolis acceptance
                            NP(xp1,y) = 1; NP(x,y) = 0; % nanoparticle moves right
                            LQ(x,y) = 1; LQ(xp1,y) = 0; % liquid moves left
                        end

                    elseif (d == 3) && (LQ(x,ym1) == 1)

                        % change in energy if nanoparticle moves down
                        dE = -(LQ(xm1,y) + LQ(x,yp1) + LQ(xp1,y) ...
                            - LQ(xm1,ym1) - LQ(x,ym2) - LQ(xp1,ym1)) ...
                            -e_nl * (LQ(xm1,ym1) + LQ(x,ym2) + LQ(xp1,ym1) ...
                            - LQ(xm1,y) - LQ(x,yp1) - LQ(xp1,y)) ...
                            -e_nl * (NP(xm1,y) + NP(x,yp1) + NP(xp1,y) ...
                            - NP(xm1,ym1) - NP(x,ym2) - NP(xp1,ym1)) ...
                            -e_nn * (NP(xm1,ym1) + NP(x,ym2) + NP(xp1,ym1) ...
                            - NP(xm1,y) - NP(x,yp1) - NP(xp1,y));

                        if r < exp(-B*dE) % Metropolis acceptance
                            NP(x,ym1) = 1; NP(x,y) = 0; % nanoparticle moves down
                            LQ(x,y) = 1; LQ(x,ym1) = 0; % liquid moves up 
                        end

                    elseif (d == 4) && (LQ(x,yp1) == 1)

                        % change in energy if nanoparticle moves up
                        dE = -(LQ(xm1,y) + LQ(x,ym1) + LQ(xp1,y) ...
                            - LQ(xm1,yp1) - LQ(x,yp2) - LQ(xp1,yp1)) ...
                            -e_nl * (LQ(xm1,yp1) + LQ(x,yp2) + LQ(xp1,yp1) ...
                            - LQ(xm1,y) - LQ(x,ym1) - LQ(xp1,y)) ...
                            -e_nl * (NP(xm1,y) + NP(x,ym1) + NP(xp1,y) ...
                            - NP(xm1,yp1) - NP(x,yp2) - NP(xp1,yp1)) ...
                            -e_nn * (NP(xm1,yp1) + NP(x,yp2) + NP(xp1,yp1) ...
                            - NP(xm1,y) - NP(x,ym1) - NP(xp1,y));

                        if r < exp(-B*dE) % Metropolis acceptance
                            NP(x,yp1) = 1; NP(x,y) = 0; % nanoparticle moves up
                            LQ(x,y) = 1; LQ(x,yp1) = 0; % liquid moves down 
                        end

                    end

                end

            end % end of nanoparticle diffusion loop

           
        end
                fname = "ImagesOriginalMatlabOrigSize/rabani_kT=" + num2str(kT, 3) + "mu="  + num2str(mu, 3) + ".png";
        
        img = (2*NP+LQ)+1;
        imwrite(img, cm, fname);
    end
end

disp(toc)