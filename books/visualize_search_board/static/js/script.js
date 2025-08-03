// 各盤面の現在の手番を管理
const boardStates = {};

// 無限スクロール用の状態管理
let isLoading = false;
let hasMoreBoards = false; // HTMLから設定される
let currentOffset = 10; // HTMLから設定される

// 初期状態の設定
function initializeBoardStates() {
    document.querySelectorAll('.board-card').forEach(card => {
        const boardIndex = parseInt(card.dataset.boardIndex);
        const moveCount = parseInt(card.querySelector('.move-count').textContent.match(/\d+/)[0]);
        boardStates[boardIndex] = {
            currentMoveIndex: -1, // 初期局面から開始
            totalMoves: moveCount
        };

        // 初期局面の表示を更新
        updateInitialDisplay(card, boardIndex);
    });
}

// 初期局面の表示を更新
function updateInitialDisplay(card, boardIndex) {
    const currentMoveElement = card.querySelector('.current-move');
    currentMoveElement.textContent = '初期局面';

    const currentIndexElement = card.querySelector('.current-index');
    currentIndexElement.textContent = '初期';

    // ボタンの状態を更新（初期局面では戻るボタンは無効）
    const buttons = card.querySelectorAll('.nav-button');
    buttons[0].disabled = true;  // first
    buttons[1].disabled = true;  // prev
    buttons[2].disabled = false; // next
    buttons[3].disabled = false; // last

    // 手順リストの現在位置をハイライト
    updateMoveListHighlight(card, -1);
}

// 手順リストのハイライトを更新
function updateMoveListHighlight(card, currentMoveIndex) {
    const moveItems = card.querySelectorAll('.move-list li');
    moveItems.forEach((item, index) => {
        item.classList.remove('current-move-item');
        if (index === currentMoveIndex) {
            item.classList.add('current-move-item');
            // 現在の手が見えるようにスクロール
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
}

// ボタンクリックイベント
document.addEventListener('click', function (e) {
    if (e.target.classList.contains('nav-button')) {
        const card = e.target.closest('.board-card');
        const boardIndex = parseInt(card.dataset.boardIndex);
        const action = e.target.dataset.action;
        handleNavigation(boardIndex, action, card);
        return;
    }

    // 手順リストのクリックイベント
    if (e.target.closest('.move-list li')) {
        const moveItem = e.target.closest('.move-list li');
        const card = e.target.closest('.board-card');
        const boardIndex = parseInt(card.dataset.boardIndex);
        const moveIndex = parseInt(moveItem.dataset.moveIndex);

        handleMoveClick(boardIndex, moveIndex, card);
        return;
    }
});

// スクロール時の無限読み込み
function handleScroll() {
    if (isLoading || !hasMoreBoards) {
        console.log('スクロール処理スキップ:', { isLoading, hasMoreBoards });
        return;
    }

    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const threshold = 100;

    console.log('スクロール位置:', {
        scrollTop: scrollTop,
        windowHeight: windowHeight,
        documentHeight: documentHeight,
        距離: documentHeight - (scrollTop + windowHeight),
        threshold: threshold
    });

    // ページ下部まで100px以内になったら読み込み開始
    if (scrollTop + windowHeight >= documentHeight - threshold) {
        console.log('読み込み閾値に到達！');
        loadMoreBoardsAuto();
    }
}

// スロットル関数（スクロールイベントの頻度を制限）
function throttle(func, limit) {
    let inThrottle;
    return function () {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

async function loadMoreBoardsAuto() {
    if (isLoading || !hasMoreBoards) {
        console.log('読み込み処理スキップ:', { isLoading, hasMoreBoards });
        return;
    }

    console.log('盤面の自動読み込み開始:', { currentOffset });
    isLoading = true;
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
        console.log('ローディングインジケーター表示');
    }

    try {
        const url = `/api/boards?offset=${currentOffset}&limit=10`;
        console.log('APIリクエスト:', url);
        const response = await fetch(url);
        const data = await response.json();

        console.log('APIレスポンス:', data);

        if (response.ok) {
            const boardsGrid = document.querySelector('.boards-grid');

            // 新しい盤面カードを追加
            data.boards.forEach(board => {
                const boardCard = createBoardCard(board);
                boardsGrid.appendChild(boardCard);

                // 新しく追加された盤面の状態を初期化
                boardStates[board.index] = {
                    currentMoveIndex: -1,
                    totalMoves: board.move_count
                };
            });

            // 状態を更新
            currentOffset = data.loaded;
            hasMoreBoards = data.has_more;

            console.log('状態更新:', { currentOffset, hasMoreBoards });

            // ヘッダーの表示を更新
            const header = document.querySelector('.header p');
            header.textContent = `${data.loaded} / ${data.total} の盤面を表示中`;

            // ローディングインジケーターを非表示
            if (loadingIndicator) {
                if (!hasMoreBoards) {
                    loadingIndicator.remove();
                    console.log('全盤面読み込み完了');
                } else {
                    loadingIndicator.style.display = 'none';
                }
            }

            console.log(`${data.boards.length} 盤面を自動読み込みしました`);
        } else {
            console.error('APIエラー:', data.error);
        }
    } catch (error) {
        console.error('Network error:', error);
    } finally {
        isLoading = false;
        console.log('読み込み処理完了');
    }
}

function createBoardCard(board) {
    const cardHTML = `
        <div class="board-card" data-board-index="${board.index}">
            <div class="board-header">
                <div class="board-title">盤面 ${board.board_index}</div>
                <div class="move-count">${board.move_count} 手</div>
            </div>

            <div class="board-content">
                <div class="board-svg">
                    ${board.board_svg}
                </div>

                <div class="board-info">
                    <div class="move-preview">
                        <h4>手順一覧</h4>
                        <ul class="move-list">
                            ${board.move_preview.map((move, index) =>
        `<li data-move-index="${index}">${move}</li>`
    ).join('')}
                        </ul>
                    </div>
                </div>
            </div>

            <div class="controls">
                <button class="nav-button" data-action="first" title="初期局面">|◀</button>
                <button class="nav-button" data-action="prev" title="前の手">◀</button>
                <div class="move-info">
                    <div class="current-move">初期局面</div>
                    <div><span class="current-index">初期</span> / ${board.move_count}</div>
                </div>
                <button class="nav-button" data-action="next" title="次の手">▶</button>
                <button class="nav-button" data-action="last" title="最終局面">▶|</button>
            </div>
        </div>
    `;

    const wrapper = document.createElement('div');
    wrapper.innerHTML = cardHTML;
    return wrapper.firstElementChild;
}

async function handleMoveClick(boardIndex, moveIndex, card) {
    const state = boardStates[boardIndex];

    // 既に同じ手番の場合は何もしない
    if (moveIndex === state.currentMoveIndex) return;

    try {
        card.classList.add('loading');

        const response = await fetch(`/api/board/${boardIndex}/${moveIndex}`);
        const data = await response.json();

        if (response.ok) {
            updateBoardDisplay(card, data, boardIndex);
        } else {
            console.error('Error:', data.error);
        }
    } catch (error) {
        console.error('Network error:', error);
    } finally {
        card.classList.remove('loading');
    }
}

async function handleNavigation(boardIndex, action, card) {
    const state = boardStates[boardIndex];
    let newMoveIndex = state.currentMoveIndex;

    switch (action) {
        case 'first':
            newMoveIndex = -1;
            break;
        case 'prev':
            newMoveIndex = Math.max(-1, state.currentMoveIndex - 1);
            break;
        case 'next':
            newMoveIndex = Math.min(state.totalMoves - 1, state.currentMoveIndex + 1);
            break;
        case 'last':
            newMoveIndex = state.totalMoves - 1;
            break;
    }

    if (newMoveIndex === state.currentMoveIndex) return;

    try {
        card.classList.add('loading');

        const response = await fetch(`/api/board/${boardIndex}/${newMoveIndex}`);
        const data = await response.json();

        if (response.ok) {
            updateBoardDisplay(card, data, boardIndex);
        } else {
            console.error('Error:', data.error);
        }
    } catch (error) {
        console.error('Network error:', error);
    } finally {
        card.classList.remove('loading');
    }
}

function updateBoardDisplay(card, data, boardIndex) {
    // 盤面SVGを更新
    const svgContainer = card.querySelector('.board-svg');
    svgContainer.innerHTML = data.board_svg;

    // 現在の手情報を更新
    const currentMoveElement = card.querySelector('.current-move');
    currentMoveElement.textContent = data.current_move;

    const currentIndexElement = card.querySelector('.current-index');
    currentIndexElement.textContent = data.move_index === -1 ? '初期' : (data.move_index + 1);

    // ボタンの状態を更新
    const buttons = card.querySelectorAll('.nav-button');
    buttons[0].disabled = !data.has_prev; // first
    buttons[1].disabled = !data.has_prev; // prev
    buttons[2].disabled = !data.has_next; // next
    buttons[3].disabled = !data.has_next; // last

    // 手順リストのハイライトを更新
    updateMoveListHighlight(card, data.move_index);

    // 状態を更新
    boardStates[boardIndex].currentMoveIndex = data.move_index;
}

// キーボードショートカット（フォーカスされたカードに対して）
document.addEventListener('keydown', function (e) {
    const focusedCard = document.querySelector('.board-card:hover') ||
        document.querySelector('.board-card:focus-within');
    if (!focusedCard) return;

    const boardIndex = parseInt(focusedCard.dataset.boardIndex);

    switch (e.key) {
        case 'ArrowLeft':
            e.preventDefault();
            handleNavigation(boardIndex, 'prev', focusedCard);
            break;
        case 'ArrowRight':
            e.preventDefault();
            handleNavigation(boardIndex, 'next', focusedCard);
            break;
        case 'Home':
            e.preventDefault();
            handleNavigation(boardIndex, 'first', focusedCard);
            break;
        case 'End':
            e.preventDefault();
            handleNavigation(boardIndex, 'last', focusedCard);
            break;
    }
});

// ページ読み込み完了時に初期化
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOMContentLoaded - 初期化開始');

    initializeBoardStates();

    // HTMLのデータ属性から無限スクロール設定を取得
    const scrollData = document.getElementById('scroll-data');
    if (scrollData) {
        hasMoreBoards = scrollData.dataset.hasMoreBoards === 'true';
        currentOffset = parseInt(scrollData.dataset.currentOffset);

        console.log('無限スクロール初期化:', {
            hasMoreBoards: hasMoreBoards,
            currentOffset: currentOffset
        });

        // スクロールイベントを設定
        if (hasMoreBoards) {
            console.log('スクロールイベントリスナーを追加');
            window.addEventListener('scroll', throttle(handleScroll, 200));
            console.log('初期化完了');
        } else {
            console.log('hasMoreBoardsがfalseのため、スクロールイベント未設定');
        }
    } else {
        console.log('scroll-data要素が見つかりません');
    }
});
