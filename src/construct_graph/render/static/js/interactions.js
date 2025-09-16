// interactions.js
// Click highlighting logic and binding.

import { appState } from './state.js';
import { fadeAllExcept, clearHighlight } from './highlight.js';
import { renderNodeDetails, renderRelationshipDetailsFromEdge, resetDetailsPanel } from './details.js';

export function bindClickHighlight() {
  const { network, nodes, edges } = appState;
  if (!network || !nodes || !edges) return;

  window.networkClickHandler = function(params) {
    // Normalize pointer pick
    let nodeAtPointer = null, edgeAtPointer = null;
    try {
      if (params && params.pointer && params.pointer.DOM) {
        nodeAtPointer = network.getNodeAt(params.pointer.DOM);
        edgeAtPointer = network.getEdgeAt(params.pointer.DOM);
      }
    } catch (e) {
      console.warn('Error in pointer detection:', e);
    }
    
    // Ensure params has the expected structure
    if (!params) {
      console.warn('No params provided to click handler');
      return;
    }
    
    if (nodeAtPointer) { 
      params.nodes = [nodeAtPointer]; 
      params.edges = []; 
    } else if (edgeAtPointer) { 
      params.nodes = []; 
      params.edges = [edgeAtPointer]; 
    }

    if (!nodeAtPointer && !edgeAtPointer) { 
      clearHighlight(); 
      resetDetailsPanel();
      return; 
    }

    if (params.nodes && params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      // Highlight triads if this node participates
      const modEdgesForEndpoint = edges.get().filter(e => e.moderatorInfo && (e.to === nodeId || e.from === nodeId));
      const medEdgesForEndpoint = edges.get().filter(e => e.mediatorInfo && (e.to === nodeId || e.from === nodeId));
      if (modEdgesForEndpoint.length > 0) {
        const nodesToHighlight = new Set([nodeId]);
        const edgeIdsToHighlight = new Set();
        modEdgesForEndpoint.forEach(me => {
          const mi = me.moderatorInfo;
          nodesToHighlight.add(mi.moderator); nodesToHighlight.add(mi.source); nodesToHighlight.add(mi.target);
          edges.get().forEach(e => { if (e.moderatorInfo && e.moderatorInfo.moderator === mi.moderator && e.moderatorInfo.source === mi.source && e.moderatorInfo.target === mi.target) edgeIdsToHighlight.add(e.id); });
          edges.get().forEach(e => { if (e.from === mi.source && e.to === mi.target) edgeIdsToHighlight.add(e.id); });
        });
        fadeAllExcept(Array.from(nodesToHighlight), Array.from(edgeIdsToHighlight), 0.1);
        renderNodeDetails(nodeId);
      } else if (medEdgesForEndpoint.length > 0) {
        const nodesToHighlight = new Set([nodeId]);
        const edgeIdsToHighlight = new Set();
        medEdgesForEndpoint.forEach(me => {
          const mi = me.mediatorInfo;
          nodesToHighlight.add(mi.mediator); nodesToHighlight.add(mi.source); nodesToHighlight.add(mi.target);
          edges.get().forEach(e => { if (e.mediatorInfo && e.mediatorInfo.mediator === mi.mediator && e.mediatorInfo.source === mi.source && e.mediatorInfo.target === mi.target) edgeIdsToHighlight.add(e.id); });
          edges.get().forEach(e => { if (e.from === mi.source && e.to === mi.target) edgeIdsToHighlight.add(e.id); });
        });
        fadeAllExcept(Array.from(nodesToHighlight), Array.from(edgeIdsToHighlight), 0.1);
        renderNodeDetails(nodeId);
      } else {
        const incident = (typeof network.getConnectedEdges === 'function') ? (network.getConnectedEdges(nodeId) || []) : [];
        const neighborList = (typeof network.getConnectedNodes === 'function') ? (network.getConnectedNodes(nodeId) || []) : [];
        const neighbors = new Set(neighborList); neighbors.add(nodeId);
        fadeAllExcept(Array.from(neighbors), incident, 0.1);
        renderNodeDetails(nodeId);
      }
      return;
    }

    if (params.edges && params.edges.length > 0) {
      const edgeId = params.edges[0];
      const e = edges.get(edgeId);
      if (!e) return;
      if (e.moderatorInfo) {
        const mi = e.moderatorInfo; const nodesToHighlight = [mi.moderator, mi.source, mi.target]; const edgeIds = [];
        edges.get().forEach(ed => { if ((ed.from === mi.source && ed.to === mi.target) || (ed.from === mi.target && ed.to === mi.source)) edgeIds.push(ed.id); });
        edges.get().forEach(ed => { if (ed.moderatorInfo && ed.moderatorInfo.moderator === mi.moderator && ed.moderatorInfo.source === mi.source && ed.moderatorInfo.target === mi.target) edgeIds.push(ed.id); });
        fadeAllExcept(nodesToHighlight, edgeIds, 0.1);
        renderRelationshipDetailsFromEdge(e);
      } else if (e.mediatorInfo) {
        const mi = e.mediatorInfo; const nodesToHighlight = [mi.mediator, mi.source, mi.target]; const edgeIds = [];
        edges.get().forEach(ed => { if ((ed.from === mi.source && ed.to === mi.target) || (ed.from === mi.target && ed.to === mi.source)) edgeIds.push(ed.id); });
        edges.get().forEach(ed => { if (ed.mediatorInfo && ed.mediatorInfo.mediator === mi.mediator && ed.mediatorInfo.source === mi.source && ed.mediatorInfo.target === mi.target) edgeIds.push(ed.id); });
        fadeAllExcept(nodesToHighlight, edgeIds, 0.1);
        renderRelationshipDetailsFromEdge(e);
      } else {
        const highlightNodes = [e.from, e.to].filter(Boolean);
        const highlightEdges = [edgeId];
        fadeAllExcept(highlightNodes, highlightEdges, 0.1);
        renderRelationshipDetailsFromEdge(e);
      }
    }
  };

  network.on('click', window.networkClickHandler);
}


