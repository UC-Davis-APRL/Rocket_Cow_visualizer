import pypropep as ppp
import streamlit as st
import pandas as pd
import numpy as np
import scipy as sp 
# import skimpy
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# import matplotlib.pyplot as plt
from scipy import constants

ppp.init()
state = st.session_state

# initialize
if 'run_button' not in state:
    state.run_button = False

if 'plot_button' not in state:
    state.plot_button = False
    

def run_button():
    state.run_button = True  

def plot_button():
    state.plot_button = True

def pypropep_to_dataframe(p, ox, fuel):

    Parameters = ['of (wt ratio)', 'p (psi)', 't (K)', 'rho (kg/m^3)', 'v (m/s)', 'Isp (s)', 'Ivac (m/s)', 'c* (m/s)', 'cf', 'sound (m/s)', 'A/At', 'cp (kJ/kg-K)', 'cv (kJ/kg-K)', 'gamma', 'mol mass (g/mol)', 
              'h (kJ/kg)', 'u (kJ/kg)', 'g (kJ/kg)', 's (kJ/kg-K)', 'dV_P', 'dV_T', 'composition']
    positions = ['chamber', 'throat', 'exit']

    df = pd.DataFrame(columns=Parameters, index=positions, dtype=float)

    df.attrs = {'ox' : [ox.formula(), ox['name'], ox['id']], 
                'fuel' : [fuel.formula(), fuel['name'], fuel['id']]}

    for i, c in enumerate(positions):
        composition = p.composition[c][0:8]
        if(bool(p.composition_condensed[c])):
            composition.append(p.composition_condensed[c])

        df.loc[c, 'of (wt ratio)']      = p._equil_structs[0].propellant.coef[1] * ox.mw
        df.loc[c, 'p (psi)']            = p._equil_objs[i].properties.P
        df.loc[c, 't (K)']              = p._equil_objs[i].properties.T
        df.loc[c, 'rho (kg/m^3)']       = (p._equil_objs[i].properties.P * 101325 * p._equil_objs[i].properties.M / 1000) / (p._equil_objs[i].properties.T * 8.314 ) # rho (kg/m^3) = (P (atm) * 101325 (Pa) / 1 (atm) * M (g/mol) * 1 kg/1000 g)/( T (K) * R (m^3-Pa/mol-k))
        df.loc[c, 'v (m/s)']            = p._equil_structs[i].performance.Isp
        df.loc[c, 'Isp (s)']            = p._equil_structs[i].performance.Isp/constants.g
        df.loc[c, 'Ivac (m/s)']         = p._equil_structs[i].performance.Ivac
        df.loc[c, 'c* (m/s)']           = p._equil_structs[i].performance.cstar
        df.loc[c, 'cf']                 = p._equil_structs[i].performance.cf
        df.loc[c, 'sound (m/s)']        = p._equil_structs[i].performance.ae_at
        df.loc[c, 'A/At']               = p._equil_objs[i].properties.Vson
        df.loc[c, 'cp (kJ/kg-K)']       = p._equil_objs[i].properties.Cp
        df.loc[c, 'cv (kJ/kg-K)']       = p._equil_objs[i].properties.Cv
        df.loc[c, 'gamma']              = p._equil_objs[i].properties.Isex
        df.loc[c, 'mol mass (g/mol)']   = p._equil_objs[i].properties.M
        df.loc[c, 'h (kJ/kg)']          = p._equil_objs[i].properties.H
        df.loc[c, 'u (kJ/kg)']          = p._equil_objs[i].properties.U
        df.loc[c, 'g (kJ/kg)']          = p._equil_objs[i].properties.G
        df.loc[c, 's (kJ/kg-K)']        = p._equil_objs[i].properties.S
        df.loc[c, 'dV_P']               = p._equil_objs[i].properties.dV_P
        df.loc[c, 'dV_T']               = p._equil_objs[i].properties.dV_T
        # df.at[c, 'composition']         = [composition]

    return df

@st.cache_data
def ranged_sim(ox, fuel, of_arr, p_arr , p_e = 1, assumption = 'SHIFTING'):

    # iterates through OF ratios and pressures. 
    df_list = [] 
    o = ppp.PROPELLANTS[ox]
    f = ppp.PROPELLANTS[fuel]

    if assumption == 'SHIFTING':
        for p in p_arr:
            for of in of_arr:
                # print(p, of)
                performance = ppp.ShiftingPerformance()
                performance.add_propellants_by_mass([(f, 1.0), (o, of)])
                performance.set_state(P=p, Pe=p_e)
                df = pypropep_to_dataframe(performance, o, f)
                df_list.append(df)
        
    elif assumption == 'FROZEN':
        for p in p_arr:
            for of in of_arr:
                # # print(p, of)
                # print(f)
                performance = ppp.FrozenPerformance()
                performance.add_propellants_by_mass([(f, 1.0), (o, of)])
                performance.set_state(P = p, Pe = p_e)
                # print(performance)
                df = pypropep_to_dataframe(performance, o, f)
                df_list.append(df)    
                  
    else: 
        raise Exception('invalid assumption, options are \'SHIFTING\' or \'FROZEN\'')
        
    results = pd.concat(df_list, keys = list(range(len(df_list)))) 
    
    # if not a list of dataframes is output
    return results

@st.cache_data
def data_filter(df, pos, var):
    # Parameters = ['of (wt ratio)', 'p (psi)', 't (K)', 'rho (kg/m^3)', 'v (m/s)', 'Isp (s)', 'Ivac (m/s)', 'c* (m/s)', 'cf', 'sound (m/s)', 'A/At', 'cp (kJ/kg-K)', 'cv (kJ/kg-K)', 'gamma', 'mol mass (g/mol)', 
    #           'h (kJ/kg)', 'u (kJ/kg)', 'g (kJ/kg)', 's (kJ/kg-K)', 'dV_P', 'dV_T', 'composition']
    # positions = ['chamber', 'throat', 'exit']
    # filtered = df.loc[:, [  'of (wt ratio)','p (psi)', var]]
    core = df.loc[pd.IndexSlice[:, 'chamber'], ['of (wt ratio)', 'p (psi)']]
    core.index = core.index.droplevel(1)
    param = df.loc[pd.IndexSlice[:, pos], var]
    param.index = param.index.droplevel(1)
    core[var] = param

    output = pd.pivot(core, index= 'p (psi)',  columns = 'of (wt ratio)')
    output = output.iloc[::-1]
    output = output.droplevel(0, 1)
    output.index.name = None
    output.columns.name = None
    output.columns = ['{:.1f}'.format(x) for x in output.columns.round(2)]
    output.attrs = core.attrs

    return output

@st.cache_data
def gen_plot(df, pos, var, plot_type = 'heatmap',): 
    data_filtered = data_filter(df, pos, var)
    print(df.attrs['ox'][0])
    if plot_type == 'heatmap': 
        fig = px.imshow(data_filtered, width=600, height=600, origin='lower', color_continuous_scale='viridis', 
                        labels={"x": "OF Ratio (%weight)", "y": "Pressure (atm)", "hover": var,'color':var}, 
                        title="{} of {} & {} Engine at the {}".format(var, df.attrs['ox'][1], df.attrs['fuel'][1], pos), aspect="auto")
    elif plot_type == 'surface':
        surface = go.Surface(z=data_filtered.values, x = data_filtered.columns.values , y = data_filtered.index.values, colorscale = 'Viridis')
        fig = go.Figure(data = [surface])
    else: 
        raise Exception('invalid plot type, options are \'heatmap\' or \'surface\'')
    
    return fig, data_filtered

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


### Creates the UI
st.header('Rocket Cow Engine Visualizer')
st.write('Welcome to V0 of the :rainbow[Rocket Cow Engine Visualizer\u2122], enter a range of pressure and OF ratio and this app will let you visualize the conditions throughout the engine at any given point')



# col1, col2 = st.columns(2)
            # with col1: 
            #     st.header("Pressure")
            # with col2:
            #     st.selectbox('', ['Psi', 'Bar', 'Mpa', 'Atm'], key='p_unit')


with st.form('OF & Pressure form'):
    # Prop input
    st.subheader("Propellants")
    col1, col2 = st.columns(2)
    col1.selectbox('Fuel:', ['RP-1 (RPL)', 'METHANE', 'PROPANE'], key='fuel')
    col2.selectbox('Oxidizer:',['OXYGEN (LIQUID)', 'OXYGEN (GAS)', 'NITROUS OXIDE', 'AIR (DRY AT SEA LEVEL)'], key='ox')
    
    # Pressure input
    st.divider()
    st.subheader("Pressure")

    col1, col2, col3 = st.columns([0.38,0.38,0.24   ])
    col1.number_input('Min Chamber Pressure (atm):', key='p_min', min_value=0.0, value=5.0, step=5.0)
    col2.number_input('Max Pressure (atm):', key='p_max', min_value = 0.0, value = 101.0,  step = 5.0)
    col3.number_input('Step Size (atm):', key='p_step', min_value=0.01, value = 5.0, step = 1.0)  

    st.number_input('Ambient Pressure (atm):', key='p_e', min_value=0.0 ,value=1.0 ,step=0.1)

    # OF conditions
    st.divider()
    st.subheader("OF Ratio")
    
    col1, col2, col3 = st.columns(3)
    col1.number_input('Min OF (%wt ratio):', key='of_min', min_value = 0.01, value = 0.5, step=0.1)
    col2.number_input('Max OF (%wt ratio):', key='of_max', min_value = 0.01, value = 10.1,  step = 0.1)
    col3.number_input('Step Size (%wt ratio):', key='of_step', min_value = 0.0001, value= 0.2, step= 0.05)
    
    # sim type and run
    st.divider()  
    st.subheader("Assumptions")

    st.selectbox('reacting flow condition', ['FROZEN', 'SHIFTING'], key='assume')
    st.form_submit_button('Run', use_container_width = True, on_click=run_button)


if state.run_button:
    
    if(state.p_min > state.p_max):
        st.warning('Your minimum pressure is greater than your maximum pressure! Adjust and rerun.', icon="⚠️")
        st.stop()
    if(state.p_min < state.p_e):
        st.warning('Your your exit pressure exceeds your minimum pressure! Your chamber pressure must always exceed ambient pressure. Adjust and rerun.', icon="⚠️")
        st.stop()
    if(state.p_step > (state.p_max-state.p_min)):
        st.warning('Your pressure step size is greater than your pressure range! Adjust and rerun', icon="⚠️")
        st.stop()

    if(state.of_min > state.of_max):
        st.warning('Your minimum OF ratio is greater than your maximum OF ratio! Adjust and rerun', icon="⚠️")
        st.stop()
    if(state.of_step > (state.p_max-state.p_min)):
        st.warning('Your OF ratio step size is greater than your OF ratio range! Adjust and rerun', icon="⚠️")
        st.stop()


    
    state.data = ranged_sim(state.ox,state.fuel , list(np.arange(state.of_min, state.of_max, state.of_step)), list(np.arange(state.p_min,state.p_max, state.p_step)), assumption = state.assume, p_e = state.p_e)
    
    st.header('Results')
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(state.data)
        csv = convert_df(state.data)
        st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')

    st.subheader('Visualization')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox('Position:', ['chamber', 'throat', 'exit'], key='plot_pos')

    with col2:
        st.selectbox('Variable:', ['t (K)', 'rho (kg/m^3)', 'v (m/s)', 'Isp (s)', 'Ivac (m/s)', 'c* (m/s)', 
                                            'cf', 'sound (m/s)', 'A/At', 'cp (kJ/kg-K)', 'cv (kJ/kg-K)', 'gamma', 'mol mass (g/mol)', 'h (kJ/kg)', 
                                            'u (kJ/kg)', 'g (kJ/kg)', 's (kJ/kg-K)', 'dV_P', 'dV_T'], key='plot_var')

    with col3:
        st.selectbox('Plot Type:', ['heatmap', 'surface'], key='plot_type')
    
    st.button('Plot', use_container_width=True, on_click=plot_button)

    if state.plot_button:
        state.figure, state.data_filtered = gen_plot(state.data, state.plot_pos, state.plot_var, plot_type=state.plot_type)
        st.plotly_chart(state.figure)

        if st.checkbox('Show plotted data'):
            st.subheader('Plotted data')
            st.dataframe(state.data_filtered)
            csv = convert_df(state.data_filtered)
            st.download_button("Press to Download", csv, "file.csv", "text/csv")    

# running stuff
# data = ranged_sim(state.ox,state.fuel , list(np.arange(state.of_min, state.of_max, state.of_step)), 
#                   list(np.arange(state.p_min,state.p_max, state.p_step)), assumption='FROZEN')
# figure = gen_plot(data, 'chamber', 't (K)', plot_type='surface')





# plot_chart(figure)
