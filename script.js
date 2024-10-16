const loginForm = document.getElementById('login-form');
const loginPage = document.getElementById('login-page');
const selectionPage = document.getElementById('selection-page');
const normalPage = document.getElementById('normal-page');
const diseasedPage = document.getElementById('diseased-page');
const recommendationResult = document.getElementById('recommendation-result');
const diseaseRecommendationResult = document.getElementById('disease-recommendation-result');
const backBtn = document.getElementById('back-btn');

let pageHistory = [];

function showPage(pageToShow) {
    const currentPage = [...document.querySelectorAll('.page')].find(page => getComputedStyle(page).display === 'block');
    if (currentPage && currentPage !== loginPage) {
        pageHistory.push(currentPage);
    }

    [loginPage, selectionPage, normalPage, diseasedPage].forEach(page => {
        page.style.display = 'none';
    });
    pageToShow.style.display = 'block';
    anime({
        targets: pageToShow,
        opacity: [0, 1],
        translateY: [20, 0],
        easing: 'easeOutCubic',
        duration: 500
    });

    backBtn.style.display = pageHistory.length > 0 ? 'flex' : 'none';
}

backBtn.addEventListener('click', () => {
    if (pageHistory.length > 0) {
        const previousPage = pageHistory.pop();
        showPage(previousPage);
    }
});

loginForm.addEventListener('submit', function (event) {
    event.preventDefault();
    showPage(selectionPage);
});

document.getElementById('normal-btn').addEventListener('click', () => showPage(normalPage));
document.getElementById('diseased-btn').addEventListener('click', () => showPage(diseasedPage));

document.getElementById('normal-submit').addEventListener('click', function () {
    const mealType = document.getElementById('meal-type').value;
    const foodType = document.getElementById('food-type').value;
    const personType = document.getElementById('person-type').value;
    const dietType = document.getElementById('diet-type').value;
    let recommendation = `Based on your ${mealType} preference for ${foodType} food, considering your ${personType} lifestyle and ${dietType} goals, we recommend a balanced meal rich in nutrients and tailored to your needs.`;
    animateResult(recommendationResult, recommendation);
});

document.getElementById('disease-submit').addEventListener('click', function () {
    const disease = document.getElementById('disease').value;
    let diseaseRecommendation = `For managing ${disease}, we recommend a carefully planned diet that supports your health needs while ensuring proper nutrition. Consult with a healthcare professional for personalized advice.`;
    animateResult(diseaseRecommendationResult, diseaseRecommendation);
});

function animateResult(element, text) {
    element.style.opacity = 0;
    element.textContent = text;
    anime({
        targets: element,
        opacity: 1,
        translateY: [20, 0],
        easing: 'easeOutCubic',
        duration: 500
    });
}

// Initial animation
anime({
    targets: '.container',
    scale: [0.9, 1],
    opacity: [0, 1],
    easing: 'easeOutCubic',
    duration: 800
});

// Button hover effect
document.querySelectorAll('button').forEach(button => {
    button.addEventListener('mouseenter', () => {
        anime({
            targets: button,
            scale: 1.05,
            duration: 300,
            easing: 'easeOutCubic'
        });
    });
    button.addEventListener('mouseleave', () => {
        anime({
            targets: button,
            scale: 1,
            duration: 300,
            easing: 'easeOutCubic'
        });
    });
});
