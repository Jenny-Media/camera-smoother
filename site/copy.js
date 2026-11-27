if (navigator.clipboard?.writeText) {
  for (const button of document.querySelectorAll('[data-copy]')) {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const text = document.getElementById(button.dataset.copy);
      const feedback = button.closest('section').querySelector('[role="status"]');
      try {
        await navigator.clipboard.writeText(text.textContent);
        feedback.textContent = 'Copied.';
      } catch {
        feedback.textContent = 'Select and copy the text above.';
      }
    });
  }
}
