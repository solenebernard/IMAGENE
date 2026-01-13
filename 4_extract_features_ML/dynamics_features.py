# Import local tools
from tools.read_yaml import *
from tools.imports import *
from tools.tools_bf import *

from scipy.stats import linregress
from scipy.spatial import KDTree, distance
from skimage.measure import regionprops_table
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point, LinearRing, box
from shapely.ops import clip_by_rect
from shapely.ops import nearest_points
import powerlaw


def TMSD(traj, dn):
    a0 = traj[:-dn]
    a1 = traj[dn:]
    dist = np.linalg.norm(a1-a0,axis=-1)
    # print(len(dist))
    return(np.mean(dist))

def TMSD_curve(traj, nmax=20):
    tau = np.arange(1,nmax)
    a = np.array([TMSD(traj, dn) for dn in tau])
    return(tau, a)

def fit_powerlaw(log_tau, log_msd):
    # Fit log-log MSD
    slope, intercept, r_value, p_value, std_err = linregress(log_tau, log_msd)
    return(slope, std_err)


def voronoi_finite_polygons(vor, labels, bbox):
    """Return finite polygons clipped to a bounding box."""
    polygons_dict = {}
    for i, region_index in enumerate(vor.point_region[:]):
        cell_bf = labels[i] # region_index value goes from 1 to max+1
        region = vor.regions[region_index] 
        if -1 in region or len(region) == 0:
            # infinite region → create a big polygon then clip
            continue
        poly = Polygon(vor.vertices[region])
        clipped = poly.intersection(box(*bbox))
        
        # Get distance to closest boundary
        pol_ext = LinearRing(clipped.exterior.coords)
        point = Point(vor.points[i])
        d = pol_ext.project(point)
        p = pol_ext.interpolate(d)
        closest_point_coords = np.asarray(p.coords)[0]
        di = np.linalg.norm(closest_point_coords-np.asarray(point.coords.xy).flatten())
        
        polygons_dict[cell_bf] = {'vor_area':clipped.area, 'distance_boundary':di}
         
        # p1, p2 = nearest_points(clipped, Point(vor.points[i]))
        # x,y = clipped.exterior.coords.xy
        # fig, ax = plt.subplots(1,1,figsize=(10,10))
        # ax.plot(x,y)
        # plt.scatter(point.coords.xy[0],point.coords.xy[1])
        # ax.scatter(closest_point_coords[0], closest_point_coords[1])
        # ax.set_aspect('equal')
        # plt.show()
    return polygons_dict

def create_df(list_masks):
    # Generate the centroids of cells
    props_all = []
    for t in range(n_t):
        props_cells = regionprops_table(
            list_masks[t],
            properties=('label', 'area', 'centroid')
        )
        df_cells = pd.DataFrame(props_cells)
        df_cells['t'] = t
        props_all.append(df_cells)
    df_cells_all = pd.concat([c for c in props_all])
    df_cells_all = df_cells_all.rename(columns={'label':'cell_bf'})
    df_cells_all = df_cells_all.set_index(['cell_bf', 't'])
    return(df_cells_all)

def create_df_traj(df_cells_all):
    # Generate metrics for dynamics
    unique_id = np.unique(np.asarray([x[0] for x in df_cells_all.index]))
    dict_traj = {}
    for id_cell in unique_id:
        traj = np.asarray(df_cells_all.loc[id_cell,['centroid-0', 'centroid-1']])
        if len(traj)>15:
            tau, msd = TMSD_curve(traj, nmax=15) 
            # Fit log-log MSD
            log_tau, log_msd = np.log10(tau), np.log10(msd)
            slope, std_err = fit_powerlaw(log_tau, log_msd)
        else:
            slope, std_err= None, None
        
        dict_traj[id_cell] = {'alpha': slope, \
                'std_err': std_err}   
    df = pd.DataFrame(dict_traj).T
    df.index.name = 'cell_bf'
    return(df)


def create_df_vor_area(df_cells_all):
    df_vor_areas_all = pd.DataFrame()
    # For each t frame
    for t in range(n_t):
        points = df_cells_all.loc[:, t, :]
        points = points[points['area']>500] # filter the error
        vor = Voronoi(points[['centroid-0','centroid-1']])
        list_bf_id = np.asarray(points.index)
        # compute voronoi area
        vor_areas = voronoi_finite_polygons(vor, list_bf_id, bbox)
        # append to the dataframe
        df_vor_areas = pd.DataFrame(vor_areas).T
        df_vor_areas.index.name = 'cell_bf'
        df_vor_areas['t'] = t
        df_vor_areas_all = pd.concat((df_vor_areas_all,df_vor_areas))
    df_vor_areas_all.set_index(['t'], inplace=True, append=True)
    return(df_vor_areas_all)

def create_df_k_neighbors(df_cells_all):
    # For each t frame
    df_all = pd.DataFrame()
    for t in range(n_t):
        points = df_cells_all.loc[:, t, :]
        df = pd.DataFrame([], index=points.index)
        pi = points[['centroid-0', 'centroid-1']]
        d = distance.cdist(pi, pi)
        list_thresh = [50,100,150,200,250]
        for thresh in list_thresh:
            n_d = np.array([np.sum(x<thresh)-1 for x in d]) # count number of neighbors closer than threshold
            df[f'n_neigh_thresh_{thresh}'] = n_d
        df['t']=t
        df_all = pd.concat((df_all, df))
    df_all.set_index('t', append=True, inplace=True)
    return(df_all)


list_masks = load_masks(ROOTPATH, EXP_PATH, CONDITION, clean=False, t_max=21)
n_t = len(list_masks)
bbox = [0, list_masks.shape[1], list_masks.shape[2], 0]

df_cells_all = create_df(list_masks)
df_traj = create_df_traj(df_cells_all)
df_vor_area = create_df_vor_area(df_cells_all)
df_k_neigh = create_df_k_neighbors(df_cells_all)

df = df_cells_all.join(df_vor_area).join(df_k_neigh)
df.to_csv(ROOTPATH + EXP_PATH + 'live_cell_features/neighbors_feats_'+CONDITION+'.csv')
df_traj.to_csv(ROOTPATH + EXP_PATH + 'live_cell_features/dynamics_feats_'+CONDITION+'.csv')