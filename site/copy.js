if (navigator.clipboard?.writeText) {
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const content = document.getElementById(button.dataset.copy);
      const feedback = button.closest('.commands').querySelector('.copy-status');
      try {
        await navigator.clipboard.writeText(content.textContent);
        feedback.textContent = 'Copied.';
      } catch {
        feedback.textContent = 'Select and copy the text above.';
      }
    });
  }
}
