function tata()

    m = biorbd('new', '/home/lim/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod');
    Qinit = zeros(biorbd('nq', m), 1);
    Qfinal = fmincon(@(x)obj(x, m), Qinit, [], [], [], [], [-pi/8, -pi/2, -pi/4, -pi/2, -0.1, -pi/4, -pi, -pi/4],[0.1, 0.1, pi, pi/2, pi, pi/4, pi, pi/4],@(x)nonlcon(x, m));
    
    hold on
    mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 11, 12, 13, 14, 15, 16, 17, 12];
    T = biorbd('markers', m, Qinit);
    jcs = biorbd('globaljcs', m, Qinit);
    cor = squeeze(jcs(1:3, 4, mapping));
    cor = [cor(:, 1:9) T(:, 19), cor(:, 9:end)];
    plot3d(T, 'k.')
    plot3d(cor, 'k')
    
    T = biorbd('markers', m, Qfinal);
    jcs = biorbd('globaljcs', m, Qfinal);
    cor = squeeze(jcs(1:3, 4, mapping));
    cor = [cor(:, 1:9) T(:, 19), cor(:, 9:end)];
    plot3d(T, 'r.')
    plot3d(cor, 'r')
    axis equal
    
end

function out = obj(Q, m)
    T = biorbd('markers', m, Q);

    out = (T(:, 17) - T(:, 36))' * (T(:, 17) - T(:, 36));
    
    
end

function [c, ceq] = nonlcon(Q, m)

    c = [];
    T = biorbd('markers', m, Q);
    A = T(:,19) - T(:,17);
    B = T(:,37) - T(:,35); 
    ceq = cross(A,B)
    ceq(4) = (T(:, 19) - T(:, 36))' * (T(:, 19) - T(:, 36));
    

end