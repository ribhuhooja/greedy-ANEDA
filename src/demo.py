import osmnx as ox
import folium
from folium.features import ClickForMarker
from jinja2 import Template
import requests
import webbrowser

import data_helper


if __name__ == '__main__':
    graph = data_helper.download_networkx_graph('hartford', 'drive') 

    _map = ox.folium.plot_graph_folium(graph, color='#a3dbff', weight=1.5)

    ClickForMarker._customtemplate = Template(u"""
            {% macro script(this, kwargs) %}
                var first_mark = null;
                var second_mark = null;
                var polyLines = [];

                function newMarker(e) {
                    var new_mark = L.marker().setLatLng(e.latlng).addTo({{this._parent.get_name()}});
                    new_mark.dragging.enable();
                    new_mark.on('dblclick', function(e){ 
                        delete markers[e.target._leaflet_id]
                        {{this._parent.get_name()}}.removeLayer(e.target)
                    })
                    var lat = e.latlng.lat.toFixed(4),
                        lng = e.latlng.lng.toFixed(4);
                    new_mark.bindPopup({{ this.popup }});

                    if(first_mark != null && second_mark != null) {
                        console.log('third');
                        {{this._parent.get_name()}}.removeLayer(first_mark);
                        {{this._parent.get_name()}}.removeLayer(second_mark);
                        polyLines.forEach(function(polyLine) {
                            {{this._parent.get_name()}}.removeLayer(polyLine);
                        });
                        first_mark = second_mark = null;
                        polyLines = [];
                    }
                    if(first_mark == null) {
                        console.log('first');
                        first_mark = new_mark;
                    } else if(second_mark == null) {
                        console.log('second');
                        second_mark = new_mark;
                        
                        // Add route
                        var src_param = first_mark._latlng.lat.toString() + "|" + first_mark._latlng.lng.toString();
                        var dest_param = second_mark._latlng.lat.toString() + "|" + second_mark._latlng.lng.toString();
                        var url = "http://localhost:5000/?src=" + src_param + "&dest=" +  dest_param;
                        fetch(url).then(resp => resp.json())
                            .then(data => {
                                data['route'].forEach(function(line) {
                                    var polyLine = L.polyline(line, {color: 'blue'});
                                    polyLines.push(polyLine);
                                    polyLine.addTo({{this._parent.get_name()}});
                                });
                            });
                    } else {
                        console.log('third');
                        {{this._parent.get_name()}}.removeLayer(first_mark);
                        {{this._parent.get_name()}}.removeLayer(second_mark);
                        polyLines.forEach(function(polyLine) {
                            {{this._parent.get_name()}}.removeLayer(polyLine);
                        });
                        polyLines = [];
                    }
                };
                {{this._parent.get_name()}}.on('click', newMarker);
            {% endmacro %}
            """)
    def __custom_init__(self, *args, **kwargs):
        self.__init_orig__(*args, **kwargs)
        self._template = self._customtemplate
    ClickForMarker.__init_orig__ = ClickForMarker.__init__
    ClickForMarker.__init__ = __custom_init__

    ClickForMarker().add_to(_map)

    path = '../demo/public/index.html'
    _map.save(path)