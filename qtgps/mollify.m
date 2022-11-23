% function [M_sig,G] = mollify(real_space, sig, M, VoxelV)
function [M_sig,G] = mollify(real_space, sig, M)


x = real_space.x
y = real_space.y
z = real_space.z

orig_size = real_space.size

dx = x(2) - x(1)
dy = y(2) - y(1)
dz = z(2) - z(1)

% Lx = x(end) + dx - x(1)
% Ly = y(end) + dy - y(1)
% Lz = z(end) + dz - z(1)
Lx = x(end) - x(1)
Ly = y(end) - y(1)
Lz = z(end) - z(1)

centerx = floor(Lx/2)
centery = floor(Ly/2)
centerz = floor(Lz/2)

% sigma_x = sig(1)
% sigma_y = sig(2)
% sigma_z = sig(3)

nx = orig_size(1)
ny = orig_size(2)
nz = orig_size(3)
G2 = zeros(orig_size)

% for i = 1:nx
%     for j = 1:ny
%         for k = 1:nz
%             xi = x(i)   yj = y(j)   zk = z(k)
%             if ((abs(xi-centerx) < sig) && ...
%                 (abs(yj-centery) < sig) &&...
%                 (abs(zk-centerz) < sig))
%                 G2(i,j,k) = exp(1/(((xi-centerx)/sig)^2 - 1)) * ...
%                             exp(1/(((yj-centery)/sig)^2 - 1)) * ...
%                             exp(1/(((zk-centerz)/sig)^2 - 1))
%             end
% % 
% %             if (norm([(xi-centerx), (yj-centery), (zk-centerz)]) < sig)
% % % %                 G2(i,j,k) = exp((((xi-centerx)/sig)^2 + ((yj-centery)/sig)^2 + ((zk-centerz)/sig)^2 - 1)^-1) 
% %                 G2(i,j,k) = exp(1/(norm([(xi-centerx)/sig, (yj-centery)/sig, (zk-centerz)/sig])^2 - 1)) 
% % % %                 G2(i,j,k) = (1 / sig)^3 * exp( ( (xi-centerx)^2 + (yj-centery)^2 + (zk-centerz)^2 - sig_x^2 )^-1) 
% %             end
%         end 
%     end
% end

for i = 1:nx
    for j = 1:ny
        for k = 1:nz
            xi = x(i)   yj = y(j)   zk = z(k)
            if (norm([xi, yj, zk]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi, yj, zk]/sig)^2 - 1))
            elseif (norm([xi-Lx, yj-Ly, zk-Lz]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi-Lx, yj-Ly, zk-Lz]/sig)^2 - 1))
            elseif (norm([xi, yj-Ly, zk-Lz]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi, yj-Ly, zk-Lz]/sig)^2 - 1))
            elseif (norm([xi-Lx, yj, zk-Lz]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi-Lx, yj, zk-Lz]/sig)^2 - 1))
            elseif (norm([xi-Lx, yj-Ly, zk]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi-Lx, yj-Ly, zk]/sig)^2 - 1))
            elseif (norm([xi, yj, zk-Lz]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi, yj, zk-Lz]/sig)^2 - 1))
            elseif (norm([xi-Lx, yj, zk]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi-Lx, yj, zk]/sig)^2 - 1))
            elseif (norm([xi, yj-Ly, zk]) < sig) 
                G2(i,j,k) = exp(1/(norm([xi, yj-Ly, zk]/sig)^2 - 1))
            end            
        end
    end
end
G2 = (1 / sig)^3 * G2
% dvol = real_space.VoxelV/prod(orig_size)
% trpz = trapz(x,trapz(y,trapz(z,G2,3),2),1)
trpz = sum(G2(:))
G = 1/trpz * G2

% integral = trapz(x,trapz(y,trapz(z,G,3),2),1)
integral = sum(G(:))

cstr_hat = fftn(M)
G_hat = fftn(G)
M_sig = ifftn(cstr_hat.*G_hat)
M_sig(M_sig <= 1e-10) = 0
