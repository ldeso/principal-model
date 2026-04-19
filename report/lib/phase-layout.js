// Measure the actual left-gutter width of each .pc-section (the distance
// from the viewport's left edge to the body column's left edge) and
// expose it to CSS as --pc-left-gutter. We read the parent's bounding
// rect rather than .pc-section's own, because the section has a negative
// margin-left that would make its own rect.left meaningless. If the
// gutter is below a usable threshold, mark the section .pc-narrow so
// the stacked layout kicks in instead of a squashed sidebar.
(function () {
  // Below this, the sidebar can't fit the fixed 11rem label column plus
  // a usable slider track + readout, so we stack instead.
  var MIN_GUTTER = 320;
  function update() {
    document.querySelectorAll('.pc-section').forEach(function (sec) {
      var parent = sec.parentElement;
      if (!parent) return;
      var left = parent.getBoundingClientRect().left;
      if (left < MIN_GUTTER) {
        sec.classList.add('pc-narrow');
        sec.style.setProperty('--pc-left-gutter', '0px');
      } else {
        sec.classList.remove('pc-narrow');
        sec.style.setProperty('--pc-left-gutter', left + 'px');
      }
    });
  }
  var schedule = function () { requestAnimationFrame(update); };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', schedule);
  } else {
    schedule();
  }
  window.addEventListener('load', schedule);
  window.addEventListener('resize', schedule);
})();
